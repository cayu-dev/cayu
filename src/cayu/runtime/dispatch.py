from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime._diagnostics import ExceptionDiagnostic
from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.invocation import (
    SessionInvocationBinding,
    TaskExecutionSource,
    copy_session_invocation_binding,
)
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    ModelTarget,
    SessionStatusConflict,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
    require_secret_free_structured_output_spec,
)
from cayu.runtime.tasks import (
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _terminalize_claimed_task_or_detect_peer_winner,
    task_create_with_runtime_invocation,
)
from cayu.vaults import SecretRedactor

logger = logging.getLogger(__name__)
_DISPATCH_DIAGNOSTIC_MAX_BYTES = 4096


class DispatchStatus(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class DispatchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    messages: list[Message]
    dispatch_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    target: ModelTarget | None = None
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
        copied_messages = [detach_message(message) for message in value]
        if not copied_messages:
            raise ValueError("DispatchRequest messages cannot be empty.")
        return copied_messages

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "metadata")

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

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("session_id", "dispatch_id", "task_id")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)


class DispatchHandle(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    dispatch_id: str
    session_id: str
    backend: str
    status: DispatchStatus = DispatchStatus.SUBMITTED
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_handle_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "metadata")

    @field_validator("dispatch_id", "session_id", "backend", "task_id")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)


class DispatchRuntime(Protocol):
    def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
        """Run dispatched work inline and stream runtime events."""


class _DurableDispatchRuntime(DispatchRuntime, Protocol):
    """Runtime capabilities required before dispatch data becomes durable."""

    def redact_dispatch_request(self, request: DispatchRequest) -> DispatchRequest:
        """Return a request safe to cross a durable dispatch boundary."""

    def redact_json(self, value: Any) -> Any:
        """Return a JSON-compatible value safe for durable publication."""

    def redact_exception_diagnostic(
        self,
        error: BaseException,
        *,
        empty_message: str,
        nonportable_message: str,
    ) -> ExceptionDiagnostic:
        """Snapshot an exception without exposing workload secrets."""

    async def session_invocation_for_dispatch(
        self,
        session_id: str,
    ) -> SessionInvocationBinding:
        """Load trusted immutable provenance for a durable dispatch target."""


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
        durable_runtime = _require_dispatch_redaction_boundary(runtime)
        if request.loop_policies:
            # loop_policies are process-local callables excluded from JSON serialization, so
            # they cannot cross a durable queue. Reject rather than silently drop them (which
            # would make a queued dispatch run with weaker guards than the inline dispatcher).
            raise ValueError(
                "TaskStoreDispatcher cannot queue a DispatchRequest with loop_policies; "
                "they are process-local and do not survive serialization."
            )
        request = _runtime_redact_dispatch_request(durable_runtime, request)
        # No defensive copy here: app.dispatch already copied the request, model_dump produces
        # an isolated snapshot, and the handle reads only immutable string fields.
        # The queue task must be session-unbound (``session_id is None``) to be claimable by
        # a worker pool; the target session_id rides inside the serialized request payload.
        session_binding = await _load_dispatch_session_invocation(
            durable_runtime,
            request.session_id,
        )
        task_request = task_create_with_runtime_invocation(
            TaskCreate(
                type=self._task_type,
                parent_task_id=request.task_id,
                input={"request": request.model_dump(mode="json")},
            ),
            source=TaskExecutionSource.TASK_DISPATCH,
            session_invocation=session_binding,
        )
        task = await self._tasks.create_task(task_request)
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
        durable_runtime = _require_dispatch_redaction_boundary(runtime)
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        task = await self._tasks.claim_task(
            worker_id,
            # FIFO: claim the oldest pending dispatch so steady arrivals can't starve it.
            TaskQuery(type=self._task_type, order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=self._lease_seconds,
        )
        if task is None:
            return None
        # Fail malformed or unauthenticated queue authority terminally rather than letting
        # the task be reclaimed and re-run forever. The payload can select executable work
        # only after it agrees with the immutable task and target-session provenance.
        payload = task.input.get("request")
        try:
            if type(payload) is not dict:
                raise ValueError("dispatch task request payload is not an object")
            request = DispatchRequest.model_validate(payload)
            session_binding = await _load_dispatch_session_invocation(
                durable_runtime,
                request.session_id,
            )
            _require_dispatch_task_authority(
                task,
                request=request,
                session_binding=session_binding,
                task_type=self._task_type,
            )
        except Exception as exc:
            diagnostic = _runtime_exception_diagnostic(
                durable_runtime,
                exc,
                empty_message="invalid dispatch request",
                nonportable_message=(
                    "Invalid dispatch authority contained a non-portable diagnostic."
                ),
            )
            failure_payload = _safe_runtime_diagnostic_payload(
                durable_runtime,
                diagnostic.payload_fields(),
            )
            diagnostic = None
            try:
                await self._commit_task_terminal(
                    task_id=task.id,
                    worker_id=worker_id,
                    kind=TaskTerminalKind.FAILED,
                    payload=failure_payload,
                )
            except TaskClaimLost:
                logger.warning(
                    "dispatch task %s lost its lease while rejecting invalid authority",
                    task.id,
                )
            return None

        # Heartbeat in the background so the lease survives long gaps between events (a slow
        # model/tool turn would otherwise let the lease lapse and another worker re-run it).
        # The outer try/finally keeps the heartbeat alive THROUGH terminalization — a slow
        # complete/fail/release must not let the lease expire and get the task reclaimed and
        # run a second time — and always stops it, including on CancelledError (graceful
        # worker shutdown), which neither except below catches.
        status = DispatchStatus.SUBMITTED
        heartbeat = asyncio.create_task(self._heartbeat(task.id, worker_id, durable_runtime))
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
                recovered = await self._recover_stalled_session(durable_runtime, request)
                try:
                    await self._tasks.release_task(task.id, worker_id)
                except TaskClaimLost:
                    logger.warning(
                        "dispatch %s lost its lease before conflict requeue",
                        request.dispatch_id,
                    )
                    return self._handle(
                        request,
                        DispatchStatus.SUBMITTED,
                        queue_task_id=task.id,
                        reclaimed=True,
                        recovered_session=recovered,
                    )
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task.id,
                    requeued=True,
                    recovered_session=recovered,
                )
            except Exception as exc:
                diagnostic = _runtime_exception_diagnostic(
                    durable_runtime,
                    exc,
                    empty_message="dispatch failed",
                    nonportable_message="Dispatch failed with a non-portable diagnostic.",
                )
                return await self._terminalize(
                    task.id,
                    worker_id,
                    request,
                    DispatchStatus.FAILED,
                    _safe_runtime_diagnostic_payload(
                        durable_runtime,
                        diagnostic.payload_fields(),
                    ),
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
        """Record the run's terminal outcome, guarded by lease ownership.

        If this worker lost the lease or another terminalization already won,
        preserve the authoritative record and return a handle marked
        ``reclaimed``.
        """
        try:
            kind = (
                TaskTerminalKind.FAILED
                if status is DispatchStatus.FAILED
                else TaskTerminalKind.COMPLETED
            )
            peer_terminalization_won = await self._commit_task_terminal(
                task_id=task_id,
                worker_id=worker_id,
                kind=kind,
                payload=payload,
            )
            if peer_terminalization_won:
                logger.warning(
                    "dispatch %s (%s) observed a peer terminalization winner",
                    request.dispatch_id,
                    status.value,
                )
                return self._handle(
                    request,
                    status,
                    queue_task_id=task_id,
                    reclaimed=True,
                )
        except TaskClaimLost:
            # The task is no longer ours (reclaimed / already terminalized elsewhere),
            # so do not clobber its current owner.
            logger.warning(
                "dispatch %s (%s) lost its lease before terminalizing; another worker will re-run it",
                request.dispatch_id,
                status.value,
            )
            return self._handle(request, status, queue_task_id=task_id, reclaimed=True)
        return self._handle(request, status, queue_task_id=task_id)

    async def _commit_task_terminal(
        self,
        *,
        task_id: str,
        worker_id: str,
        kind: TaskTerminalKind,
        payload: dict[str, Any],
    ) -> bool:
        return await _terminalize_claimed_task_or_detect_peer_winner(
            self._tasks,
            TaskTerminalizationRequest(
                task_id=task_id,
                worker_id=worker_id,
                kind=kind,
                result=payload if kind is TaskTerminalKind.COMPLETED else None,
                error=payload if kind is TaskTerminalKind.FAILED else None,
                idempotency_key=_dispatch_terminalization_key(
                    task_id=task_id,
                    worker_id=worker_id,
                    kind=kind,
                ),
            ),
        )

    async def _recover_stalled_session(
        self,
        runtime: _DurableDispatchRuntime,
        request: DispatchRequest,
    ) -> bool:
        """Best-effort finalization of a session stranded in a live status by a crashed worker.

        Uses the runtime's incomplete-session recovery when available while the
        durable redaction capabilities remain mandatory. The store atomically checks the
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
        except Exception as exc:
            logger.warning(
                "dispatch %s could not recover stalled session %s: error_type=%s error=%s",
                request.dispatch_id,
                request.session_id,
                type(exc).__name__,
                _safe_runtime_text(runtime, str(exc)),
            )
            return False
        return bool(_STALLED_RECOVERED_ACTIONS & set(result.actions))

    async def _heartbeat(
        self,
        task_id: str,
        worker_id: str,
        runtime: _DurableDispatchRuntime,
    ) -> None:
        """Extend the lease every ``lease_seconds / 3`` until cancelled (best effort)."""
        interval = self._lease_seconds / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await self._tasks.heartbeat(task_id, worker_id, extend_seconds=self._lease_seconds)
            except Exception as exc:
                logger.warning(
                    "dispatch heartbeat failed for task %s: error_type=%s error=%s",
                    task_id,
                    type(exc).__name__,
                    _safe_runtime_text(runtime, str(exc)),
                )

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
        durable_runtime = _require_dispatch_redaction_boundary(runtime)
        loop = asyncio.get_running_loop()
        next_reclaim = loop.time()
        while not stop.is_set():
            if loop.time() >= next_reclaim:
                try:
                    await self._tasks.reclaim_expired(query=TaskQuery(type=self._task_type))
                except Exception as exc:
                    logger.warning(
                        "dispatch reclaim_expired failed: error_type=%s error=%s",
                        type(exc).__name__,
                        _safe_runtime_text(durable_runtime, str(exc)),
                    )
                next_reclaim = loop.time() + reclaim_every_s
            try:
                handle = await self.process_next(runtime, worker_id=worker_id)
            except Exception as exc:
                # A transient store error on one task must not kill the durable worker loop.
                logger.error(
                    "dispatch worker failed while processing a task: error_type=%s error=%s",
                    type(exc).__name__,
                    _safe_runtime_text(durable_runtime, str(exc)),
                )
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
        messages=[detach_message(message) for message in request.messages],
        dispatch_id=request.dispatch_id,
        task_id=request.task_id,
        target=(
            None
            if request.target is None
            else ModelTarget(
                provider_name=request.target.provider_name,
                model=request.target.model,
            )
        ),
        metadata=copy_durable_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


async def _load_dispatch_session_invocation(
    runtime: _DurableDispatchRuntime,
    session_id: str,
) -> SessionInvocationBinding:
    try:
        invocation_loader = runtime.session_invocation_for_dispatch
    except (AttributeError, TypeError):
        raise TypeError("Durable dispatch requires session invocation provenance.") from None
    if not callable(invocation_loader):
        raise TypeError("Durable dispatch requires session invocation provenance.")
    binding = await invocation_loader(session_id)
    if not isinstance(binding, SessionInvocationBinding):
        raise TypeError("Durable dispatch returned invalid session invocation provenance.")
    return copy_session_invocation_binding(binding)


def _require_dispatch_task_authority(
    task: Task,
    *,
    request: DispatchRequest,
    session_binding: SessionInvocationBinding,
    task_type: str,
) -> None:
    if not isinstance(task, Task):
        raise TypeError("Dispatch task authority requires a Task.")
    if task.type != task_type or task.session_id is not None:
        raise ValueError("Dispatch task structural authority conflicts with its queue.")
    if task.parent_task_id != request.task_id:
        raise ValueError("Dispatch task parent authority conflicts with its request.")
    invocation = task.invocation
    target = session_binding.invocation
    if (
        invocation.source is not TaskExecutionSource.TASK_DISPATCH
        or invocation.origin != target.origin
        or invocation.root_invocation_id != target.root_invocation_id
        or invocation.root_session_id != target.root_session_id
    ):
        raise ValueError("Dispatch task invocation provenance conflicts with its target session.")


def redact_dispatch_request(
    request: DispatchRequest,
    *,
    redactor: SecretRedactor,
) -> DispatchRequest:
    """Return the executable request shape that is safe to persist in a task queue."""

    request = copy_dispatch_request(request)
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if request.loop_policies:
        raise ValueError(
            "A durable DispatchRequest cannot contain loop_policies; they are "
            "process-local and cannot be persisted without weakening execution policy."
        )
    for field_name, value in (
        ("session_id", request.session_id),
        ("dispatch_id", request.dispatch_id),
        ("task_id", request.task_id),
        (
            "target.provider_name",
            None if request.target is None else request.target.provider_name,
        ),
        ("target.model", None if request.target is None else request.target.model),
    ):
        if value is not None and redactor.redact_text(value) != value:
            raise ValueError(
                f"DispatchRequest.{field_name} contains a workload secret and cannot "
                "be used as durable dispatch authority."
            )

    redactor.require_no_secret_keys(
        request.metadata,
        field_name="DispatchRequest.metadata",
        match_short_substrings=True,
    )
    metadata = redactor.redact_json_values(request.metadata)
    if type(metadata) is not dict:
        raise AssertionError("Dispatch metadata redaction returned a non-object.")

    require_secret_free_structured_output_spec(
        request.structured_output,
        redactor=redactor,
        field_name="DispatchRequest.structured_output",
    )

    for field_name, value in (
        ("limits", request.limits.model_dump(mode="json")),
        (
            "budget_limits",
            [limit.model_dump(mode="json") for limit in request.budget_limits],
        ),
        (
            "retry_policy",
            (
                None
                if request.retry_policy is None
                else request.retry_policy.model_dump(mode="json")
            ),
        ),
        (
            "thinking",
            None if request.thinking is None else request.thinking.model_dump(mode="json"),
        ),
    ):
        if redactor.redact_json_values(value) != value:
            raise ValueError(
                f"DispatchRequest.{field_name} contains a workload secret and cannot be "
                "persisted without changing execution semantics."
            )
    for limit_index, limit in enumerate(request.budget_limits):
        for price_index, price in enumerate(limit.pricing.prices):
            pricing_context = price.pricing_context
            if pricing_context is None:
                continue
            redactor.require_no_secret_keys(
                {dimension: None for dimension in pricing_context.dimensions},
                field_name=(
                    f"DispatchRequest.budget_limits[{limit_index}].pricing."
                    f"prices[{price_index}].pricing_context.dimensions"
                ),
                match_short_substrings=True,
            )

    return DispatchRequest(
        session_id=request.session_id,
        messages=[
            redact_untrusted_message_for_boundary(
                message,
                redactor=redactor,
                field_name="DispatchRequest.messages",
            )
            for message in request.messages
        ],
        dispatch_id=request.dispatch_id,
        task_id=request.task_id,
        target=(
            None
            if request.target is None
            else ModelTarget(
                provider_name=request.target.provider_name,
                model=request.target.model,
            )
        ),
        metadata=metadata,
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=(
            copy_retry_policy(request.retry_policy) if request.retry_policy is not None else None
        ),
        structured_output=request.structured_output,
        thinking=request.thinking,
        loop_policies=(),
    )


def _safe_runtime_text(runtime: _DurableDispatchRuntime, value: str) -> str:
    redacted = _runtime_redact_json(runtime, value)
    if type(redacted) is not str:
        raise TypeError("Dispatch runtime string redaction returned a non-string.")
    encoded = redacted.encode("utf-8", "replace")
    if len(encoded) <= _DISPATCH_DIAGNOSTIC_MAX_BYTES:
        return redacted
    marker = b"...[truncated]"
    return (encoded[: _DISPATCH_DIAGNOSTIC_MAX_BYTES - len(marker)] + marker).decode(
        "utf-8",
        "ignore",
    )


def _runtime_exception_diagnostic(
    runtime: _DurableDispatchRuntime,
    error: BaseException,
    *,
    empty_message: str,
    nonportable_message: str,
) -> ExceptionDiagnostic:
    """Snapshot a dispatch failure through the runtime's redactor before bounding."""

    diagnostic = runtime.redact_exception_diagnostic(
        error,
        empty_message=empty_message,
        nonportable_message=nonportable_message,
    )
    if type(diagnostic) is not ExceptionDiagnostic:
        raise TypeError(
            "Dispatch runtime redact_exception_diagnostic must return ExceptionDiagnostic."
        )
    return diagnostic


def _safe_runtime_diagnostic_payload(
    runtime: _DurableDispatchRuntime,
    payload: dict[str, Any],
) -> dict[str, Any]:
    redacted = _runtime_redact_json(runtime, payload)
    payload.clear()
    if type(redacted) is not dict:
        raise TypeError("Dispatch runtime diagnostic redaction returned a non-object.")
    return {
        key: _safe_runtime_text(runtime, value) if type(value) is str else value
        for key, value in redacted.items()
    }


def _runtime_redact_json(runtime: _DurableDispatchRuntime, value: Any) -> Any:
    return runtime.redact_json(value)


def _runtime_redact_dispatch_request(
    runtime: _DurableDispatchRuntime,
    request: DispatchRequest,
) -> DispatchRequest:
    redacted = runtime.redact_dispatch_request(request)
    if type(redacted) is not DispatchRequest:
        raise TypeError("Dispatch runtime request redaction returned an invalid request.")
    return copy_dispatch_request(redacted)


def _require_dispatch_redaction_boundary(
    runtime: DispatchRuntime,
) -> _DurableDispatchRuntime:
    """Reject runtimes that cannot make durable dispatch publication secret-safe."""

    for method_name in (
        "redact_dispatch_request",
        "redact_json",
        "redact_exception_diagnostic",
    ):
        try:
            method = getattr(runtime, method_name)
        except (AttributeError, TypeError):
            raise TypeError(f"Dispatch runtime {method_name} must be callable.") from None
        if not callable(method):
            raise TypeError(f"Dispatch runtime {method_name} must be callable.")
    return cast("_DurableDispatchRuntime", runtime)


def copy_dispatch_handle(handle: DispatchHandle) -> DispatchHandle:
    if type(handle) is not DispatchHandle:
        raise TypeError("Dispatch handle copy requires a DispatchHandle.")
    return DispatchHandle(
        dispatch_id=handle.dispatch_id,
        session_id=handle.session_id,
        task_id=handle.task_id,
        backend=handle.backend,
        status=handle.status,
        metadata=copy_durable_json_value(handle.metadata, "metadata"),
    )


def _dispatch_terminalization_key(
    *,
    task_id: str,
    worker_id: str,
    kind: TaskTerminalKind,
) -> str:
    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.dispatch-task-terminalization.v1",
            "task_id": task_id,
            "worker_id": worker_id,
            "kind": kind.value,
        },
        "dispatch_task_terminalization",
    )
    return f"dispatch-task-terminal:v1:{sha256(identity).hexdigest()}"


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
