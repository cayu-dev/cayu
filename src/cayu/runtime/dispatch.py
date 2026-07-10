from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, copy_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    SessionStatusConflict,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.runtime.tasks import TaskCreate, TaskOrder, TaskQuery, TaskStore

logger = logging.getLogger(__name__)


class DispatchStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class DispatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    messages: list[Message]
    dispatch_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        copied_messages = [copy_message(message) for message in value]
        if not copied_messages:
            raise ValueError("DispatchRequest messages cannot be empty.")
        return copied_messages

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("session_id", "dispatch_id", "task_id", "model")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class DispatchHandle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str
    session_id: str
    backend: str
    status: DispatchStatus = DispatchStatus.SUBMITTED
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_handle_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("dispatch_id", "session_id", "backend", "task_id")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class DispatchRuntime(Protocol):
    def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
        """Run dispatched work inline and stream runtime events."""


class Dispatcher(ABC):
    """Execution backend for dispatched session work."""

    @abstractmethod
    async def submit(
        self,
        runtime: DispatchRuntime,
        request: DispatchRequest,
    ) -> DispatchHandle:
        """Submit dispatched session work and return a handle."""


class InlineDispatcher(Dispatcher):
    """Runs dispatched session work immediately in the current process."""

    backend = "inline"

    async def submit(
        self,
        runtime: DispatchRuntime,
        request: DispatchRequest,
    ) -> DispatchHandle:
        request = copy_dispatch_request(request)
        status = DispatchStatus.SUBMITTED
        event_count = 0
        async for event in runtime.dispatch_inline(request):
            event_count += 1
            status = _dispatch_status_after_event(event, fallback=status)
        return DispatchHandle(
            dispatch_id=request.dispatch_id,
            session_id=request.session_id,
            task_id=request.task_id,
            backend=self.backend,
            status=status,
            metadata={"events": event_count},
        )


DEFAULT_DISPATCH_TASK_TYPE = "cayu.dispatch"
DISPATCH_CONFLICT_RECOVERY_REASON = "dispatch_conflict_worker_crash_recovery"

_STALLED_RECOVERED_ACTIONS = {
    IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
    IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,
    IncompleteSessionRecoveryAction.PENDING_APPROVAL,
    IncompleteSessionRecoveryAction.PENDING_USER_INPUT,
}


class TaskStoreDispatcher(Dispatcher):
    """Queue-backed dispatcher that persists work as claimable tasks in a ``TaskStore``.

    ``submit`` enqueues a ``DispatchRequest`` as a PENDING task instead of running it; a
    worker process claims it (atomically — ``PostgresTaskStore`` uses ``FOR UPDATE SKIP
    LOCKED``) and runs it through ``dispatch_inline``. Works with any ``TaskStore`` tier:
    ``InMemoryTaskStore`` (single process), ``SQLiteTaskStore`` (single node), or
    ``PostgresTaskStore`` (a distributed worker pool). Callers interact through
    ``DispatchHandle``/``DispatchStatus``; the backing Task id is surfaced as
    ``metadata["queue_task_id"]`` for observability.
    """

    backend = "task_store"

    def __init__(
        self,
        task_store: TaskStore,
        *,
        task_type: str = DEFAULT_DISPATCH_TASK_TYPE,
        lease_seconds: int = 300,
        recover_stalled_sessions_after_seconds: int | None = None,
    ) -> None:
        if not isinstance(task_store, TaskStore):
            raise TypeError("TaskStoreDispatcher requires a TaskStore.")
        if type(lease_seconds) is not int or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer.")
        if recover_stalled_sessions_after_seconds is not None and (
            type(recover_stalled_sessions_after_seconds) is not int
            or recover_stalled_sessions_after_seconds < 0
        ):
            raise ValueError(
                "recover_stalled_sessions_after_seconds must be a non-negative integer."
            )
        self._tasks = task_store
        self._task_type = require_clean_nonblank(task_type, "task_type")
        self._lease_seconds = lease_seconds
        # Horizon after which a conflicting live-status session is considered stranded
        # by a crashed worker (defaults to the task lease: a healthy run whose lease
        # would already have expired is treated the same as a crashed one).
        self._recover_stalled_after_seconds = (
            lease_seconds
            if recover_stalled_sessions_after_seconds is None
            else recover_stalled_sessions_after_seconds
        )

    async def submit(
        self,
        runtime: DispatchRuntime,
        request: DispatchRequest,
    ) -> DispatchHandle:
        if request.loop_policies:
            # loop_policies are process-local callables excluded from JSON serialization, so
            # they cannot cross a durable queue. Reject rather than silently drop them (which
            # would make a queued dispatch run with weaker guards than the inline dispatcher).
            raise ValueError(
                "TaskStoreDispatcher cannot queue a DispatchRequest with loop_policies; "
                "they are process-local and do not survive serialization."
            )
        # No defensive copy here: app.dispatch already copied the request, model_dump produces
        # an isolated snapshot, and the handle reads only immutable string fields.
        # The queue task must be session-unbound (``session_id is None``) to be claimable by
        # a worker pool; the target session_id rides inside the serialized request payload.
        task = await self._tasks.create_task(
            TaskCreate(
                type=self._task_type,
                parent_task_id=request.task_id,
                input={"request": request.model_dump(mode="json")},
            )
        )
        return self._handle(request, DispatchStatus.SUBMITTED, queue_task_id=task.id)

    async def process_next(
        self,
        runtime: DispatchRuntime,
        *,
        worker_id: str,
    ) -> DispatchHandle | None:
        """Claim and run one queued dispatch.

        Returns ``None`` if the queue is empty, or if the claimed task's payload was
        malformed (in which case the task is failed before returning).
        """
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        task = await self._tasks.claim_task(
            worker_id,
            # FIFO: claim the oldest pending dispatch so steady arrivals can't starve it.
            TaskQuery(type=self._task_type, order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=self._lease_seconds,
        )
        if task is None:
            return None
        # Fail a malformed payload (missing or no-longer-valid request — e.g. an older
        # serialization claimed after a schema change) terminally, rather than letting the
        # error escape and leave the task to be reclaimed and re-fail forever.
        payload = task.input.get("request")
        try:
            if type(payload) is not dict:
                raise ValueError("dispatch task request payload is not an object")
            request = DispatchRequest.model_validate(payload)
        except Exception as exc:
            await self._tasks.fail_task(
                task.id,
                {
                    "error": f"invalid dispatch request payload: {exc}",
                    "error_type": type(exc).__name__,
                },
                worker_id=worker_id,
            )
            return None

        # Heartbeat in the background so the lease survives long gaps between events (a slow
        # model/tool turn would otherwise let the lease lapse and another worker re-run it).
        # The outer try/finally keeps the heartbeat alive THROUGH terminalization — a slow
        # complete/fail/release must not let the lease expire and get the task reclaimed and
        # run a second time — and always stops it, including on CancelledError (graceful
        # worker shutdown), which neither except below catches.
        status = DispatchStatus.SUBMITTED
        heartbeat = asyncio.create_task(self._heartbeat(task.id, worker_id))
        try:
            try:
                async for event in runtime.dispatch_inline(request):
                    status = _dispatch_status_after_event(event, fallback=status)
            except SessionStatusConflict:
                # The session is already being run by another worker — requeue rather than
                # fail, so it runs once that session frees up (per-session serialization).
                # After a worker crash, though, the session is stranded in a live status
                # forever and every re-claim of the reclaimed task would conflict in a
                # loop; recover a stalled session so the requeued dispatch can proceed.
                recovered = await self._recover_stalled_session(runtime, request)
                await self._tasks.release_task(task.id, worker_id)
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task.id,
                    requeued=True,
                    recovered_session=recovered,
                )
            except Exception as exc:
                return await self._terminalize(
                    task.id,
                    worker_id,
                    request,
                    DispatchStatus.FAILED,
                    {"error": str(exc), "error_type": type(exc).__name__},
                )
            # A run can fail in-band (a SESSION_FAILED event, not an exception); record that as
            # a failed task so failure queries and retries see it, not a COMPLETED one.
            return await self._terminalize(
                task.id, worker_id, request, status, {"status": status.value}
            )
        finally:
            await self._stop_heartbeat(heartbeat)

    async def _terminalize(
        self,
        task_id: str,
        worker_id: str,
        request: DispatchRequest,
        status: DispatchStatus,
        payload: dict[str, Any],
    ) -> DispatchHandle:
        """Record the run's terminal outcome, guarded by lease ownership. If this worker lost
        the lease (the task was reclaimed by another worker), don't clobber its record — log
        and return a handle marked ``reclaimed``; the reclaiming worker re-runs it."""
        try:
            if status is DispatchStatus.FAILED:
                await self._tasks.fail_task(task_id, payload, worker_id=worker_id)
            else:
                await self._tasks.complete_task(task_id, payload, worker_id=worker_id)
        except ValueError:
            # Only the ownership/lease guard can raise ValueError here; it means the task is no
            # longer ours (reclaimed / already terminalized elsewhere), so we must not clobber.
            logger.warning(
                "dispatch %s (%s) lost its lease before terminalizing; another worker will re-run it",
                request.dispatch_id,
                status.value,
            )
            return self._handle(request, status, queue_task_id=task_id, reclaimed=True)
        return self._handle(request, status, queue_task_id=task_id)

    async def _recover_stalled_session(
        self,
        runtime: DispatchRuntime,
        request: DispatchRequest,
    ) -> bool:
        """Best-effort finalization of a session stranded in a live status by a crashed worker.

        Uses the runtime's incomplete-session recovery when available (duck-typed so the
        ``DispatchRuntime`` protocol stays minimal). The store atomically checks the
        durable activity horizon and increments the run epoch before recovery, so a
        genuinely live run is left alone and an evicted worker cannot write after the
        decision. Returns True when the session was recovered out of its stranded status.
        """
        recover = getattr(runtime, "recover_incomplete_session", None)
        if recover is None:
            return False
        try:
            inactive_before = datetime.now(UTC) - timedelta(
                seconds=self._recover_stalled_after_seconds
            )
            result = await recover(
                IncompleteSessionRecoveryRequest(
                    session_id=request.session_id,
                    inactive_before=inactive_before,
                    reason=DISPATCH_CONFLICT_RECOVERY_REASON,
                    metadata={"dispatch_id": request.dispatch_id},
                )
            )
        except Exception:
            logger.warning(
                "dispatch %s could not recover stalled session %s",
                request.dispatch_id,
                request.session_id,
                exc_info=True,
            )
            return False
        return bool(_STALLED_RECOVERED_ACTIONS & set(result.actions))

    async def _heartbeat(self, task_id: str, worker_id: str) -> None:
        """Extend the lease every ``lease_seconds / 3`` until cancelled (best effort)."""
        interval = self._lease_seconds / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await self._tasks.heartbeat(task_id, worker_id, extend_seconds=self._lease_seconds)
            except Exception:
                logger.warning("dispatch heartbeat failed for task %s", task_id, exc_info=True)

    @staticmethod
    async def _stop_heartbeat(heartbeat: asyncio.Task[None]) -> None:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    async def run_worker(
        self,
        runtime: DispatchRuntime,
        *,
        worker_id: str,
        stop: asyncio.Event,
        poll_interval_s: float = 1.0,
        reclaim_every_s: float = 60.0,
    ) -> None:
        """Claim-and-run loop until ``stop`` is set, periodically reclaiming dead leases."""
        loop = asyncio.get_running_loop()
        next_reclaim = loop.time()
        while not stop.is_set():
            if loop.time() >= next_reclaim:
                try:
                    await self._tasks.reclaim_expired(query=TaskQuery(type=self._task_type))
                except Exception:
                    logger.warning("dispatch reclaim_expired failed", exc_info=True)
                next_reclaim = loop.time() + reclaim_every_s
            try:
                handle = await self.process_next(runtime, worker_id=worker_id)
            except Exception:
                # A transient store error on one task must not kill the durable worker loop.
                logger.exception("dispatch worker failed while processing a task")
                handle = None
            # Back off when idle, after a busy-session requeue, or after a lost-lease reclaim —
            # otherwise the just-released/reclaimed task (FIFO-oldest) is re-claimed immediately
            # in a tight loop, re-running the agent with no delay.
            if (
                handle is None
                or handle.metadata.get("requeued")
                or handle.metadata.get("reclaimed")
            ):
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)

    def _handle(
        self,
        request: DispatchRequest,
        status: DispatchStatus,
        *,
        queue_task_id: str,
        requeued: bool = False,
        reclaimed: bool = False,
        recovered_session: bool = False,
    ) -> DispatchHandle:
        metadata: dict[str, Any] = {"queue_task_id": queue_task_id}
        if requeued:
            metadata["requeued"] = True
        if reclaimed:
            metadata["reclaimed"] = True
        if recovered_session:
            metadata["recovered_session"] = True
        return DispatchHandle(
            dispatch_id=request.dispatch_id,
            session_id=request.session_id,
            task_id=request.task_id,
            backend=self.backend,
            status=status,
            metadata=metadata,
        )


def copy_dispatch_request(request: DispatchRequest) -> DispatchRequest:
    if type(request) is not DispatchRequest:
        raise TypeError("Dispatch requires a DispatchRequest.")
    return DispatchRequest(
        session_id=request.session_id,
        messages=[copy_message(message) for message in request.messages],
        dispatch_id=request.dispatch_id,
        task_id=request.task_id,
        model=request.model,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_dispatch_handle(handle: DispatchHandle) -> DispatchHandle:
    if type(handle) is not DispatchHandle:
        raise TypeError("Dispatch handle copy requires a DispatchHandle.")
    return DispatchHandle(
        dispatch_id=handle.dispatch_id,
        session_id=handle.session_id,
        task_id=handle.task_id,
        backend=handle.backend,
        status=handle.status,
        metadata=copy_json_value(handle.metadata, "metadata"),
    )


def _dispatch_status_after_event(
    event: Event,
    *,
    fallback: DispatchStatus,
) -> DispatchStatus:
    if event.type == EventType.SESSION_RESUMED:
        return DispatchStatus.RUNNING
    if event.type == EventType.SESSION_COMPLETED:
        return DispatchStatus.COMPLETED
    if event.type == EventType.SESSION_FAILED:
        return DispatchStatus.FAILED
    if event.type == EventType.SESSION_INTERRUPTED:
        return DispatchStatus.INTERRUPTED
    return fallback
