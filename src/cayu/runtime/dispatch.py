from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
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
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    ExecutionProfileMismatchError,
    _ExecutionProfileAdmissionRequestRejected,
)
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
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    SessionRunFenced,
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
    TaskStatus,
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


_QUEUED_DISPATCH_RECORD_TYPE = "cayu.queued-dispatch"
_QUEUED_DISPATCH_SCHEMA_VERSION = 1


class _QueuedDispatchSettlementState(StrEnum):
    """Store-owned evidence available before queue-task terminalization."""

    NOT_ADMITTED = "not_admitted"
    TERMINAL_EVIDENCE_PENDING = "terminal_evidence_pending"
    TERMINAL_EVIDENCE_DURABLE = "terminal_evidence_durable"


@dataclass(frozen=True, slots=True)
class _QueuedDispatchSettlement:
    """Exact session-side authority available to a queue worker."""

    state: _QueuedDispatchSettlementState
    terminal_status: DispatchStatus | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not _QueuedDispatchSettlementState:
            raise TypeError("Queued dispatch settlement state has an invalid type.")
        if self.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE:
            if type(self.terminal_status) is not DispatchStatus:
                raise ValueError(
                    "Durable queued dispatch settlement requires an exact terminal status."
                )
        elif self.terminal_status is not None:
            raise ValueError(
                "Non-terminal queued dispatch settlement cannot carry a terminal status."
            )


class _QueuedDispatchAuthorityRejected(RuntimeError):
    """Permanent rejection proven by the runtime-owned dispatch boundary."""


class _QueuedDispatchEnvelope(BaseModel):
    """Runtime-owned authority persisted before queued work becomes claimable."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.queued-dispatch"] = _QUEUED_DISPATCH_RECORD_TYPE
    schema_version: Literal[1] = _QUEUED_DISPATCH_SCHEMA_VERSION
    queue_task_id: str
    dispatch_operation_id: str
    terminal_event_id: str
    request_sha256: str
    session_instance_fingerprint: str
    request: DispatchRequest
    source_profile: ExecutionProfileIdentity
    required_profile: ExecutionProfileIdentity

    @field_validator("queue_task_id", "terminal_event_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator(
        "dispatch_operation_id",
        "request_sha256",
        "session_instance_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> DispatchRequest:
        if isinstance(value, DispatchRequest):
            return copy_dispatch_request(value)
        return DispatchRequest.model_validate(value)

    @field_validator("source_profile", "required_profile", mode="before")
    @classmethod
    def copy_execution_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)

    @model_validator(mode="after")
    def validate_authority_tuple(self) -> _QueuedDispatchEnvelope:
        request_sha256 = _queued_dispatch_request_sha256(self.request)
        if self.request_sha256 != request_sha256:
            raise ValueError("Queued dispatch request digest does not match its request.")
        operation_id = _queued_dispatch_operation_id(
            queue_task_id=self.queue_task_id,
            request=self.request,
            request_sha256=request_sha256,
            session_instance_fingerprint=self.session_instance_fingerprint,
            source_profile=self.source_profile,
            required_profile=self.required_profile,
        )
        if self.dispatch_operation_id != operation_id:
            raise ValueError(
                "Queued dispatch operation identity conflicts with its authority tuple."
            )
        if self.terminal_event_id != _queued_dispatch_terminal_event_id(operation_id):
            raise ValueError(
                "Queued dispatch terminal event identity conflicts with its operation."
            )
        return self


class _ProfiledDispatchRuntime(_DurableDispatchRuntime, Protocol):
    """Private runtime seam for profile-bound durable dispatch."""

    async def _prepare_queued_dispatch(
        self,
        request: DispatchRequest,
        *,
        queue_task_id: str,
    ) -> _QueuedDispatchEnvelope:
        """Resolve runtime-owned profile authority before queue publication."""

    def _dispatch_queued(self, envelope: _QueuedDispatchEnvelope) -> AsyncIterator[Event]:
        """Execute one validated queued envelope under its recorded authority."""

    async def _queued_dispatch_requests_match(
        self,
        existing: DispatchRequest,
        candidate: DispatchRequest,
    ) -> bool:
        """Compare requests after resolving their authenticated session authority."""

    async def _queued_dispatch_settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        """Classify exact session-side evidence before task terminalization."""

    async def _list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery,
    ) -> list[QueuedDispatchTerminalReceipt]:
        """Discover bounded live session receipts after worker restart."""

    async def _acknowledge_queued_dispatch(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> None:
        """Release terminal-evidence retention after queue terminalization commits."""


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
    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
    IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,
    IncompleteSessionRecoveryAction.PENDING_APPROVAL,
    IncompleteSessionRecoveryAction.PENDING_USER_INPUT,
}


class TaskStoreDispatcher(Dispatcher):
    """Queue-backed dispatcher that persists work as claimable tasks in a ``TaskStore``.

    ``submit`` freezes the target session's runtime-owned execution profile and enqueues
    the resulting envelope as a PENDING task instead of running it. A worker application
    claims that task (atomically — ``PostgresTaskStore`` uses ``FOR UPDATE SKIP LOCKED``),
    validates the recorded profile, and enters the ordinary resume engine through the
    built-in queued-dispatch boundary. A bare custom ``DispatchRuntime`` does not provide
    that authority boundary; producers and workers must use a compatible ``CayuApp``
    runtime. The dispatcher works with any ``TaskStore`` tier: ``InMemoryTaskStore``
    (single process), ``SQLiteTaskStore`` (single node), or ``PostgresTaskStore`` (a
    distributed worker pool). Callers interact through ``DispatchHandle``/
    ``DispatchStatus``; the backing Task id is surfaced as
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
        self._terminal_receipt_reconciliation_cursor: tuple[str, str] | None = None
        self._terminal_receipt_reconciliation_cycle_settled = True
        self._terminal_receipt_reconciliation_task: asyncio.Task[bool] | None = None
        self._terminal_receipt_reconciliation_generation = 0
        self._startup_terminal_receipt_reconciliation_pending = True

    async def submit(
        self,
        runtime: DispatchRuntime,
        request: DispatchRequest,
    ) -> DispatchHandle:
        durable_runtime = _require_profiled_dispatch_runtime(runtime)
        if request.loop_policies:
            # loop_policies are process-local callables excluded from JSON serialization, so
            # they cannot cross a durable queue. Reject rather than silently drop them (which
            # would make a queued dispatch run with weaker guards than the inline dispatcher).
            raise ValueError(
                "TaskStoreDispatcher cannot queue a DispatchRequest with loop_policies; "
                "they are process-local and do not survive serialization."
            )
        request = _runtime_redact_dispatch_request(durable_runtime, request)
        handle_request = request
        queue_task_id = _queued_dispatch_task_id(request, task_type=self._task_type)
        existing = await self._tasks.load_task(queue_task_id)
        if existing is not None:
            envelope = _existing_queued_dispatch_envelope(
                existing,
                task_type=self._task_type,
            )
            if envelope is None or not await durable_runtime._queued_dispatch_requests_match(
                envelope.request,
                request,
            ):
                raise RuntimeError("Existing task conflicts with the queued dispatch authority.")
            settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
            session_binding = await _load_dispatch_session_invocation(
                durable_runtime,
                request.session_id,
            )
            _require_dispatch_task_authority(
                existing,
                request=envelope.request,
                session_binding=session_binding,
                task_type=self._task_type,
            )
            if existing.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                await self._acknowledge_terminal_task(durable_runtime, existing, envelope)
            return self._handle(
                handle_request,
                DispatchStatus.SUBMITTED,
                queue_task_id=existing.id,
                envelope=envelope,
                idempotent_submission=True,
            )
        session_binding = await _load_dispatch_session_invocation(
            durable_runtime,
            request.session_id,
        )
        envelope = await durable_runtime._prepare_queued_dispatch(
            request,
            queue_task_id=queue_task_id,
        )
        if type(envelope) is not _QueuedDispatchEnvelope:
            raise TypeError("Dispatch runtime returned an invalid queued dispatch envelope.")
        envelope = _copy_queued_dispatch_envelope(envelope)
        if envelope.queue_task_id != queue_task_id:
            raise ValueError("Queued dispatch envelope changed its runtime-owned task identity.")
        request = envelope.request
        # The queue task must be session-unbound (``session_id is None``) to be claimable by
        # a worker pool; the target session_id rides inside the serialized request payload.
        create_request = task_create_with_runtime_invocation(
            TaskCreate(
                task_id=queue_task_id,
                type=self._task_type,
                parent_task_id=request.task_id,
                input={"dispatch": envelope.model_dump(mode="json")},
            ),
            source=TaskExecutionSource.TASK_DISPATCH,
            session_invocation=session_binding,
        )
        idempotent_submission = False
        try:
            task = await self._tasks.create_task(create_request)
        except Exception as publication_failure:
            try:
                existing = await self._tasks.load_task(queue_task_id)
            except Exception as reconciliation_failure:
                publication_failure.add_note(
                    "Queued dispatch publication reconciliation also failed: "
                    f"{type(reconciliation_failure).__name__}."
                )
                raise publication_failure from reconciliation_failure
            existing_envelope = (
                None
                if existing is None
                else _existing_queued_dispatch_envelope(
                    existing,
                    task_type=self._task_type,
                )
            )
            if (
                existing is None
                or existing_envelope is None
                or not await durable_runtime._queued_dispatch_requests_match(
                    existing_envelope.request,
                    request,
                )
            ):
                raise
            task = existing
            envelope = existing_envelope
            request = envelope.request
            idempotent_submission = True
        if not _task_matches_queued_dispatch(
            task,
            task_type=self._task_type,
            parent_task_id=request.task_id,
            envelope=envelope,
        ):
            raise RuntimeError("Task store returned conflicting queued dispatch authority.")
        if idempotent_submission:
            settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
        _require_dispatch_task_authority(
            task,
            request=request,
            session_binding=session_binding,
            task_type=self._task_type,
        )
        if idempotent_submission and task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            await self._acknowledge_terminal_task(durable_runtime, task, envelope)
        return self._handle(
            handle_request,
            DispatchStatus.SUBMITTED,
            queue_task_id=task.id,
            envelope=envelope,
            idempotent_submission=idempotent_submission,
        )

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
        durable_runtime = _require_profiled_dispatch_runtime(runtime)
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        if self._startup_terminal_receipt_reconciliation_pending:
            reconciliation_generation = self._terminal_receipt_reconciliation_generation
            try:
                reconciliation_complete = await self._reconcile_terminal_acknowledgements(
                    durable_runtime
                )
            except Exception as exc:
                logger.warning(
                    "dispatch terminal acknowledgement discovery failed: error_type=%s error=%s",
                    type(exc).__name__,
                    _safe_runtime_text(durable_runtime, str(exc)),
                )
            else:
                self._startup_terminal_receipt_reconciliation_pending = (
                    not reconciliation_complete
                    or reconciliation_generation != self._terminal_receipt_reconciliation_generation
                )
        task = await self._tasks.claim_task(
            worker_id,
            # FIFO: claim the oldest pending dispatch so steady arrivals can't starve it.
            TaskQuery(type=self._task_type, order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=self._lease_seconds,
        )
        if task is None:
            return None
        # Fail malformed or unauthenticated queue authority terminally rather than letting
        # the task be reclaimed and re-run forever. Only the immutable task row is consulted
        # in this phase: store-backed session authority remains operational and retryable.
        payload = task.input.get("dispatch")
        try:
            if type(payload) is not dict:
                raise ValueError("dispatch task envelope payload is not an object")
            envelope = _QueuedDispatchEnvelope.model_validate(payload)
            if not _claimed_task_matches_queued_dispatch(
                task,
                task_type=self._task_type,
                worker_id=worker_id,
                envelope=envelope,
            ):
                raise ValueError("dispatch task row conflicts with its envelope")
        except (TypeError, ValueError) as exc:
            await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                error=exc,
                envelope=None,
            )
            return None

        request = envelope.request
        try:
            settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch settlement returned an invalid record."
                )
        except _QueuedDispatchAuthorityRejected as exc:
            return await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                error=exc,
                envelope=envelope,
            )
        except Exception as exc:
            await self._release_claimed_dispatch_after_failure(
                task=task,
                worker_id=worker_id,
                failure=exc,
            )
            raise

        try:
            session_binding = await _load_dispatch_session_invocation(
                durable_runtime,
                request.session_id,
            )
        except _QueuedDispatchAuthorityRejected as exc:
            return await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                error=exc,
                envelope=envelope,
            )
        except Exception as exc:
            await self._release_claimed_dispatch_after_failure(
                task=task,
                worker_id=worker_id,
                failure=exc,
            )
            raise
        try:
            _require_dispatch_task_authority(
                task,
                request=request,
                session_binding=session_binding,
                task_type=self._task_type,
            )
        except (TypeError, ValueError) as exc:
            return await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                error=exc,
                envelope=envelope,
            )

        if settlement.state is not _QueuedDispatchSettlementState.NOT_ADMITTED:
            status = settlement.terminal_status or DispatchStatus.SUBMITTED
            heartbeat = asyncio.create_task(self._heartbeat(task.id, worker_id, durable_runtime))
            try:
                return await self._terminalize(
                    durable_runtime,
                    task.id,
                    worker_id,
                    request,
                    status,
                    {"status": status.value, **_queued_dispatch_evidence(envelope)},
                    envelope=envelope,
                    settlement=settlement,
                )
            finally:
                await self._stop_heartbeat(heartbeat)

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
                async for event in durable_runtime._dispatch_queued(envelope):
                    status = _dispatch_status_after_event(event, fallback=status)
            except (SessionRunFenced, SessionStatusConflict):
                # The session is already being run by another worker — requeue rather than
                # fail, so it runs once that session frees up (per-session serialization).
                # The same rule applies while terminal hooks or trailing cleanup retain the
                # prior invocation's profile fence. After a worker crash, recover stale
                # session ownership so the requeued dispatch can proceed.
                recovered = await self._recover_stalled_session(
                    durable_runtime,
                    request,
                )
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
                        envelope=envelope,
                        reclaimed=True,
                        recovered_session=recovered,
                    )
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task.id,
                    envelope=envelope,
                    requeued=True,
                    recovered_session=recovered,
                )
            except (
                ExecutionProfileMismatchError,
                _QueuedDispatchAuthorityRejected,
            ) as exc:
                return await self._reject_claimed_dispatch(
                    durable_runtime,
                    task=task,
                    worker_id=worker_id,
                    error=exc,
                    envelope=envelope,
                )
            except Exception as exc:
                diagnostic = _queued_dispatch_failure_diagnostic(
                    durable_runtime,
                    exc,
                    empty_message="dispatch failed",
                    nonportable_message="Dispatch failed with a non-portable diagnostic.",
                )
                failure_payload = _safe_runtime_diagnostic_payload(
                    durable_runtime,
                    diagnostic.payload_fields(),
                )
                diagnostic = None
                try:
                    settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
                    if type(settlement) is not _QueuedDispatchSettlement:
                        raise _QueuedDispatchAuthorityRejected(
                            "Queued dispatch settlement returned an invalid record."
                        )
                except _QueuedDispatchAuthorityRejected as authority_error:
                    return await self._reject_claimed_dispatch(
                        durable_runtime,
                        task=task,
                        worker_id=worker_id,
                        error=authority_error,
                        envelope=envelope,
                    )
                except Exception as settlement_error:
                    combined_failure = ExceptionGroup(
                        "Queued dispatch failed before settlement could be classified.",
                        [exc, settlement_error],
                    )
                    await self._release_claimed_dispatch_after_failure(
                        task=task,
                        worker_id=worker_id,
                        failure=combined_failure,
                    )
                    raise combined_failure from None
                if settlement.state is _QueuedDispatchSettlementState.NOT_ADMITTED:
                    if isinstance(
                        exc,
                        _ExecutionProfileAdmissionRequestRejected,
                    ):
                        return await self._reject_claimed_dispatch(
                            durable_runtime,
                            task=task,
                            worker_id=worker_id,
                            error=exc,
                            envelope=envelope,
                        )
                    await self._release_claimed_dispatch_after_failure(
                        task=task,
                        worker_id=worker_id,
                        failure=exc,
                    )
                    raise
                return await self._terminalize(
                    durable_runtime,
                    task.id,
                    worker_id,
                    request,
                    DispatchStatus.FAILED,
                    {
                        **failure_payload,
                        **_queued_dispatch_evidence(envelope),
                        "status": DispatchStatus.FAILED.value,
                    },
                    envelope=envelope,
                    settlement=settlement,
                )
            # A run can fail in-band (a SESSION_FAILED event, not an exception); record that as
            # a failed task so failure queries and retries see it, not a COMPLETED one.
            return await self._terminalize(
                durable_runtime,
                task.id,
                worker_id,
                request,
                status,
                {"status": status.value, **_queued_dispatch_evidence(envelope)},
                envelope=envelope,
            )
        finally:
            await self._stop_heartbeat(heartbeat)

    async def _reject_claimed_dispatch(
        self,
        runtime: _DurableDispatchRuntime,
        *,
        task: Task,
        worker_id: str,
        error: BaseException,
        envelope: _QueuedDispatchEnvelope | None,
    ) -> DispatchHandle | None:
        """Persist a permanent rejection, with evidence only from an authenticated row."""

        diagnostic = _queued_dispatch_failure_diagnostic(
            runtime,
            error,
            empty_message="invalid dispatch request",
            nonportable_message=("Invalid dispatch authority contained a non-portable diagnostic."),
        )
        failure_payload = _safe_runtime_diagnostic_payload(
            runtime,
            diagnostic.payload_fields(),
        )
        if envelope is not None:
            failure_payload = {
                **failure_payload,
                **_queued_dispatch_evidence(envelope),
                "status": DispatchStatus.FAILED.value,
            }
        try:
            peer_terminalization_won = await self._commit_task_terminal(
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
            if envelope is None:
                return None
            authoritative_status = await self._reclaimed_dispatch_status(
                task_id=task.id,
                envelope=envelope,
            )
            return self._handle(
                envelope.request,
                authoritative_status,
                queue_task_id=task.id,
                envelope=envelope,
                reclaimed=True,
            )
        if envelope is None:
            return None
        authoritative_status = DispatchStatus.FAILED
        if peer_terminalization_won:
            authoritative_status = await self._reclaimed_dispatch_status(
                task_id=task.id,
                envelope=envelope,
            )
        return self._handle(
            envelope.request,
            authoritative_status,
            queue_task_id=task.id,
            envelope=envelope,
            reclaimed=peer_terminalization_won,
        )

    async def _reclaimed_dispatch_status(
        self,
        *,
        task_id: str,
        envelope: _QueuedDispatchEnvelope,
    ) -> DispatchStatus:
        """Return only status proven by the task that won a rejected claim."""

        current = await self._tasks.load_task(task_id)
        if current is None:
            raise RuntimeError("Queued dispatch task disappeared after its claim was lost.")
        if current.status is TaskStatus.CANCELLED:
            return DispatchStatus.CANCELLED
        if current.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            try:
                return _terminal_queued_dispatch_status(
                    current,
                    task_type=self._task_type,
                    envelope=envelope,
                )
            except RuntimeError:
                # A control-plane terminal outcome without exact dispatch evidence
                # owns the task, but does not prove a session-dispatch outcome.
                return DispatchStatus.SUBMITTED
        return DispatchStatus.SUBMITTED

    async def _release_claimed_dispatch_after_failure(
        self,
        *,
        task: Task,
        worker_id: str,
        failure: Exception,
    ) -> None:
        """Make pre-admission work reclaimable without hiding a release failure."""

        try:
            await self._tasks.release_task(task.id, worker_id)
        except TaskClaimLost:
            logger.warning(
                "dispatch task %s lost its lease while preserving a retryable failure",
                task.id,
            )
        except asyncio.CancelledError as cancellation:
            raise cancellation from failure
        except Exception as release_error:
            raise ExceptionGroup(
                "Queued dispatch pre-admission failure and task release both failed.",
                [failure, release_error],
            ) from None

    async def _terminalize(
        self,
        runtime: _ProfiledDispatchRuntime,
        task_id: str,
        worker_id: str,
        request: DispatchRequest,
        status: DispatchStatus,
        payload: dict[str, Any],
        *,
        envelope: _QueuedDispatchEnvelope,
        settlement: _QueuedDispatchSettlement | None = None,
    ) -> DispatchHandle:
        """Record the run's terminal outcome, guarded by lease ownership.

        If this worker lost the lease or another terminalization already won,
        preserve the authoritative record and return a handle marked
        ``reclaimed``.
        """
        if settlement is None:
            settlement = await runtime._queued_dispatch_settlement_state(envelope)
        if type(settlement) is not _QueuedDispatchSettlement:
            raise TypeError("Queued dispatch settlement returned an invalid record.")
        settlement_state = settlement.state
        if settlement_state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING or (
            settlement_state is _QueuedDispatchSettlementState.NOT_ADMITTED
            and status is not DispatchStatus.FAILED
        ):
            recovered = await self._recover_stalled_session(runtime, request)
            try:
                await self._tasks.release_task(task_id, worker_id)
            except TaskClaimLost:
                logger.warning(
                    "dispatch %s lost its lease while retaining incomplete terminal evidence",
                    request.dispatch_id,
                )
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task_id,
                    envelope=envelope,
                    reclaimed=True,
                    recovered_session=recovered,
                )
            return self._handle(
                request,
                DispatchStatus.SUBMITTED,
                queue_task_id=task_id,
                envelope=envelope,
                requeued=True,
                recovered_session=recovered,
            )
        if settlement_state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE:
            assert settlement.terminal_status is not None
            status = settlement.terminal_status
            payload = {**payload, "status": status.value}
        try:
            kind = (
                TaskTerminalKind.FAILED
                if status is DispatchStatus.FAILED
                else TaskTerminalKind.COMPLETED
            )
            self._arm_terminal_receipt_reconciliation()
            peer_terminalization_won = await self._commit_task_terminal(
                task_id=task_id,
                worker_id=worker_id,
                kind=kind,
                payload=payload,
            )
            terminal_task = await self._tasks.load_task(task_id)
            if terminal_task is None:
                raise RuntimeError(
                    "Queued dispatch task disappeared after terminalization committed."
                )
            authoritative_status = await self._acknowledge_terminal_task(
                runtime,
                terminal_task,
                envelope,
            )
            if peer_terminalization_won:
                logger.warning(
                    "dispatch %s (%s) observed a peer terminalization winner",
                    request.dispatch_id,
                    status.value,
                )
                return self._handle(
                    request,
                    authoritative_status,
                    queue_task_id=task_id,
                    envelope=envelope,
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
            return self._handle(
                request,
                status,
                queue_task_id=task_id,
                envelope=envelope,
                reclaimed=True,
            )
        return self._handle(
            request,
            authoritative_status,
            queue_task_id=task_id,
            envelope=envelope,
        )

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
        durable_runtime = _require_profiled_dispatch_runtime(runtime)
        loop = asyncio.get_running_loop()
        next_reclaim = loop.time()
        while not stop.is_set():
            if loop.time() >= next_reclaim:
                reconciliation_generation = self._terminal_receipt_reconciliation_generation
                try:
                    reconciliation_complete = await self._reconcile_terminal_acknowledgements(
                        durable_runtime
                    )
                    self._startup_terminal_receipt_reconciliation_pending = (
                        not reconciliation_complete
                        or reconciliation_generation
                        != self._terminal_receipt_reconciliation_generation
                    )
                except Exception as exc:
                    logger.warning(
                        "dispatch terminal acknowledgement discovery failed: "
                        "error_type=%s error=%s",
                        type(exc).__name__,
                        _safe_runtime_text(durable_runtime, str(exc)),
                    )
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

    async def _acknowledge_terminal_task(
        self,
        runtime: _ProfiledDispatchRuntime,
        task: Task,
        envelope: _QueuedDispatchEnvelope,
        *,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> DispatchStatus:
        """Release a session receipt only for an exact durable task outcome."""

        try:
            settlement = await runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
            session_binding = await _load_dispatch_session_invocation(
                runtime,
                envelope.request.session_id,
            )
            _require_dispatch_task_authority(
                task,
                request=envelope.request,
                session_binding=session_binding,
                task_type=self._task_type,
            )
            if task.status is TaskStatus.CANCELLED:
                if not _task_matches_queued_dispatch(
                    task,
                    task_type=self._task_type,
                    parent_task_id=envelope.request.task_id,
                    envelope=envelope,
                ):
                    raise RuntimeError("Cancelled queue task conflicts with its dispatch envelope.")
                if settlement.state is _QueuedDispatchSettlementState.NOT_ADMITTED:
                    if receipt is not None:
                        raise RuntimeError(
                            "Queued dispatch receipt has no exact durable terminal evidence."
                        )
                    return DispatchStatus.CANCELLED
                if (
                    settlement.state is not _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE
                    or settlement.terminal_status is None
                ):
                    raise RuntimeError(
                        "Cancelled queue task cannot release pending terminal evidence."
                    )
                dispatch_status = settlement.terminal_status
                authoritative_status = DispatchStatus.CANCELLED
            else:
                dispatch_status = _terminal_queued_dispatch_status(
                    task,
                    task_type=self._task_type,
                    envelope=envelope,
                )
                authoritative_status = dispatch_status
            if receipt is None:
                await runtime._acknowledge_queued_dispatch(
                    envelope,
                    dispatch_status=dispatch_status,
                )
            else:
                await runtime._acknowledge_queued_dispatch(
                    envelope,
                    dispatch_status=dispatch_status,
                    receipt=receipt,
                )
        except BaseException:
            self._arm_terminal_receipt_reconciliation()
            raise
        return authoritative_status

    def _arm_terminal_receipt_reconciliation(self) -> None:
        """Keep a same-process sweep pending across a terminal handoff boundary."""

        self._terminal_receipt_reconciliation_generation += 1
        self._startup_terminal_receipt_reconciliation_pending = True

    async def _reconcile_terminal_acknowledgements(
        self,
        runtime: _ProfiledDispatchRuntime,
    ) -> bool:
        """Serialize bounded cross-store acknowledgement-loss discovery."""

        while self._terminal_receipt_reconciliation_task is not None:
            active_reconciliation = self._terminal_receipt_reconciliation_task
            if active_reconciliation.get_loop() is not asyncio.get_running_loop():
                if active_reconciliation.done():
                    self._terminal_receipt_reconciliation_task = None
                    continue
                raise RuntimeError(
                    "TaskStoreDispatcher cannot reconcile terminal receipts from "
                    "multiple event loops concurrently."
                )
            await asyncio.shield(active_reconciliation)

        reconciliation = asyncio.create_task(
            self._reconcile_terminal_acknowledgements_owned(runtime)
        )
        self._terminal_receipt_reconciliation_task = reconciliation

        def reconciliation_done(completed: asyncio.Task[bool]) -> None:
            with contextlib.suppress(asyncio.CancelledError):
                completed.exception()
            if self._terminal_receipt_reconciliation_task is completed:
                self._terminal_receipt_reconciliation_task = None

        reconciliation.add_done_callback(reconciliation_done)
        return await asyncio.shield(reconciliation)

    async def _reconcile_terminal_acknowledgements_owned(
        self,
        runtime: _ProfiledDispatchRuntime,
    ) -> bool:
        """Advance one bounded, dispatcher-owned receipt reconciliation sweep."""

        page_size = 100
        max_pages = 10
        all_receipts_settled = self._terminal_receipt_reconciliation_cycle_settled
        for _ in range(max_pages):
            cursor = self._terminal_receipt_reconciliation_cursor
            after_session_id = None if cursor is None else cursor[0]
            after_operation_id = None if cursor is None else cursor[1]
            returned_receipts = await runtime._list_queued_dispatch_terminal_receipts(
                QueuedDispatchTerminalReceiptQuery(
                    after_session_id=after_session_id,
                    after_operation_id=after_operation_id,
                    limit=page_size,
                )
            )
            if type(returned_receipts) is not list or len(returned_receipts) > page_size:
                raise RuntimeError("Queued dispatch receipt discovery returned an invalid page.")
            receipts: list[QueuedDispatchTerminalReceipt] = []
            previous_key: tuple[str, str] | None = None
            if after_session_id is not None:
                assert after_operation_id is not None
                previous_key = (after_session_id, after_operation_id)
            for returned_receipt in returned_receipts:
                if type(returned_receipt) is not QueuedDispatchTerminalReceipt:
                    raise TypeError("Queued dispatch receipt discovery returned an invalid record.")
                receipt = QueuedDispatchTerminalReceipt(
                    session_id=returned_receipt.session_id,
                    queue_task_id=returned_receipt.queue_task_id,
                    operation_id=returned_receipt.operation_id,
                    terminal_event_id=returned_receipt.terminal_event_id,
                )
                key = (receipt.session_id, receipt.operation_id)
                if previous_key is not None and key <= previous_key:
                    raise RuntimeError(
                        "Queued dispatch receipt discovery did not advance its keyset."
                    )
                receipts.append(receipt)
                previous_key = key
            for receipt in receipts:
                task = await self._tasks.load_task(receipt.queue_task_id)
                if task is None:
                    all_receipts_settled = False
                    logger.error(
                        "queued dispatch task %s disappeared before terminal acknowledgement",
                        receipt.queue_task_id,
                    )
                    continue
                if task.status not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    all_receipts_settled = False
                    continue
                envelope = _existing_queued_dispatch_envelope(
                    task,
                    task_type=self._task_type,
                )
                if envelope is None or (
                    receipt.operation_id != envelope.dispatch_operation_id
                    or receipt.terminal_event_id != envelope.terminal_event_id
                ):
                    all_receipts_settled = False
                    logger.error(
                        "queued dispatch task %s conflicts with its session receipt",
                        receipt.queue_task_id,
                    )
                    continue
                try:
                    await self._acknowledge_terminal_task(
                        runtime,
                        task,
                        envelope,
                        receipt=receipt,
                    )
                except Exception as exc:
                    all_receipts_settled = False
                    logger.warning(
                        "queued dispatch task %s restart acknowledgement failed: "
                        "error_type=%s error=%s",
                        task.id,
                        type(exc).__name__,
                        _safe_runtime_text(runtime, str(exc)),
                    )
            if len(receipts) < page_size:
                self._terminal_receipt_reconciliation_cursor = None
                self._terminal_receipt_reconciliation_cycle_settled = True
                return all_receipts_settled
            last_receipt = receipts[-1]
            self._terminal_receipt_reconciliation_cursor = (
                last_receipt.session_id,
                last_receipt.operation_id,
            )
            self._terminal_receipt_reconciliation_cycle_settled = all_receipts_settled
        return False

    def _handle(
        self,
        request: DispatchRequest,
        status: DispatchStatus,
        *,
        queue_task_id: str,
        envelope: _QueuedDispatchEnvelope,
        requeued: bool = False,
        reclaimed: bool = False,
        recovered_session: bool = False,
        idempotent_submission: bool = False,
    ) -> DispatchHandle:
        metadata: dict[str, Any] = {
            "queue_task_id": queue_task_id,
            **_queued_dispatch_evidence(envelope),
        }
        if requeued:
            metadata["requeued"] = True
        if reclaimed:
            metadata["reclaimed"] = True
        if recovered_session:
            metadata["recovered_session"] = True
        if idempotent_submission:
            metadata["idempotent_submission"] = True
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
        raise _QueuedDispatchAuthorityRejected(
            "Durable dispatch requires session invocation provenance."
        ) from None
    if not callable(invocation_loader):
        raise _QueuedDispatchAuthorityRejected(
            "Durable dispatch requires session invocation provenance."
        )
    binding = await invocation_loader(session_id)
    if not isinstance(binding, SessionInvocationBinding):
        raise _QueuedDispatchAuthorityRejected(
            "Durable dispatch returned invalid session invocation provenance."
        )
    try:
        return copy_session_invocation_binding(binding)
    except (TypeError, ValueError) as exc:
        raise _QueuedDispatchAuthorityRejected(
            "Durable dispatch returned invalid session invocation provenance."
        ) from exc


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


def _queued_dispatch_request_sha256(request: DispatchRequest) -> str:
    payload = copy_dispatch_request(request).model_dump(mode="json")
    return sha256(canonical_durable_json_bytes(payload, "queued_dispatch.request")).hexdigest()


def _queued_dispatch_task_id(
    request: DispatchRequest,
    *,
    task_type: str = DEFAULT_DISPATCH_TASK_TYPE,
) -> str:
    request = copy_dispatch_request(request)
    material = {
        "schema": "cayu.queued-dispatch-task.v1",
        "task_type": require_durable_clean_nonblank(task_type, "task_type"),
        "dispatch_id": request.dispatch_id,
    }
    digest = sha256(
        canonical_durable_json_bytes(material, "queued_dispatch.task_identity")
    ).hexdigest()
    return f"cayu-dispatch-{digest}"


def _queued_dispatch_operation_id(
    *,
    queue_task_id: str,
    request: DispatchRequest,
    request_sha256: str,
    session_instance_fingerprint: str,
    source_profile: ExecutionProfileIdentity,
    required_profile: ExecutionProfileIdentity,
) -> str:
    material = {
        "record_type": _QUEUED_DISPATCH_RECORD_TYPE,
        "schema_version": _QUEUED_DISPATCH_SCHEMA_VERSION,
        "queue_task_id": require_durable_clean_nonblank(queue_task_id, "queue_task_id"),
        "dispatch_id": request.dispatch_id,
        "session_id": request.session_id,
        "linked_task_id": request.task_id,
        "request_sha256": request_sha256,
        "session_instance_fingerprint": session_instance_fingerprint,
        "source_profile_fingerprint": source_profile.fingerprint,
        "required_profile_fingerprint": required_profile.fingerprint,
    }
    return sha256(canonical_durable_json_bytes(material, "queued_dispatch.operation")).hexdigest()


def _queued_dispatch_terminal_event_id(operation_id: str) -> str:
    if len(operation_id) != 64 or any(
        character not in "0123456789abcdef" for character in operation_id
    ):
        raise ValueError("Queued dispatch operation_id must be a lowercase SHA-256 digest.")
    return f"cayu-queued-dispatch-terminal-{operation_id}"


def _new_queued_dispatch_envelope(
    *,
    queue_task_id: str,
    request: DispatchRequest,
    session_instance_fingerprint: str,
    source_profile: ExecutionProfileIdentity,
    required_profile: ExecutionProfileIdentity,
) -> _QueuedDispatchEnvelope:
    """Build one immutable envelope from runtime-owned session/profile authority."""

    request = copy_dispatch_request(request)
    request_sha256 = _queued_dispatch_request_sha256(request)
    operation_id = _queued_dispatch_operation_id(
        queue_task_id=queue_task_id,
        request=request,
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        source_profile=source_profile,
        required_profile=required_profile,
    )
    return _QueuedDispatchEnvelope(
        queue_task_id=queue_task_id,
        dispatch_operation_id=operation_id,
        terminal_event_id=_queued_dispatch_terminal_event_id(operation_id),
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        request=request,
        source_profile=source_profile,
        required_profile=required_profile,
    )


def _copy_queued_dispatch_envelope(
    envelope: _QueuedDispatchEnvelope,
) -> _QueuedDispatchEnvelope:
    if type(envelope) is not _QueuedDispatchEnvelope:
        raise TypeError("Queued dispatch requires an exact runtime envelope.")
    return _QueuedDispatchEnvelope.model_validate(envelope.model_dump(mode="json"))


def _queued_dispatch_evidence(envelope: _QueuedDispatchEnvelope) -> dict[str, str]:
    return {
        "dispatch_operation_id": envelope.dispatch_operation_id,
        "session_instance_fingerprint": envelope.session_instance_fingerprint,
        "source_execution_profile_fingerprint": envelope.source_profile.fingerprint,
        "required_execution_profile_fingerprint": envelope.required_profile.fingerprint,
    }


def _terminal_queued_dispatch_status(
    task: Task,
    *,
    task_type: str,
    envelope: _QueuedDispatchEnvelope,
) -> DispatchStatus:
    """Validate the exact terminal task evidence that authorizes receipt release."""

    if not _task_matches_queued_dispatch(
        task,
        task_type=task_type,
        parent_task_id=envelope.request.task_id,
        envelope=envelope,
    ):
        raise RuntimeError("Terminal queue task conflicts with its dispatch envelope.")
    if task.status is TaskStatus.COMPLETED:
        payload = task.result
        allowed_statuses = {
            DispatchStatus.COMPLETED,
            DispatchStatus.INTERRUPTED,
        }
    elif task.status is TaskStatus.FAILED:
        payload = task.error
        allowed_statuses = {DispatchStatus.FAILED}
    else:
        raise RuntimeError(
            "Queued dispatch acknowledgement requires an exact completed or failed task."
        )
    if type(payload) is not dict:
        raise RuntimeError("Terminal queue task has no structured dispatch outcome.")
    if (
        payload.get("dispatch_operation_id") != envelope.dispatch_operation_id
        or payload.get("session_instance_fingerprint") != envelope.session_instance_fingerprint
        or payload.get("source_execution_profile_fingerprint")
        != envelope.source_profile.fingerprint
        or payload.get("required_execution_profile_fingerprint")
        != envelope.required_profile.fingerprint
    ):
        raise RuntimeError("Terminal queue task has conflicting dispatch authority.")
    raw_status = payload.get("status")
    if type(raw_status) is not str:
        raise RuntimeError("Terminal queue task has no exact dispatch status evidence.")
    try:
        dispatch_status = DispatchStatus(raw_status)
    except ValueError:
        raise RuntimeError("Terminal queue task has invalid dispatch status evidence.") from None
    if dispatch_status not in allowed_statuses:
        raise RuntimeError("Terminal queue task status conflicts with its durable outcome.")
    return dispatch_status


def _task_matches_queued_dispatch(
    task: Task,
    *,
    task_type: str,
    parent_task_id: str | None,
    envelope: _QueuedDispatchEnvelope,
) -> bool:
    """Require complete equality before treating queue publication as replayed."""

    return (
        type(task) is Task
        and task.id == envelope.queue_task_id
        and task.id == _queued_dispatch_task_id(envelope.request, task_type=task_type)
        and task.type == task_type
        and task.session_id is None
        and task.parent_task_id == parent_task_id
        and task.title is None
        and task.description is None
        and task.assigned_agent_name is None
        and task.available_at is None
        and task.metadata == {}
        and task.input == {"dispatch": envelope.model_dump(mode="json")}
    )


def _claimed_task_matches_queued_dispatch(
    task: Task,
    *,
    task_type: str,
    worker_id: str,
    envelope: _QueuedDispatchEnvelope,
) -> bool:
    """Require the complete immutable row plus positive current-claim evidence."""

    return (
        _task_matches_queued_dispatch(
            task,
            task_type=task_type,
            parent_task_id=envelope.request.task_id,
            envelope=envelope,
        )
        and task.status is TaskStatus.CLAIMED
        and task.worker_id == worker_id
        and task.lease_expires_at is not None
    )


def _existing_queued_dispatch_envelope(
    task: Task,
    *,
    task_type: str,
) -> _QueuedDispatchEnvelope | None:
    """Return exact durable authority for an idempotent submission retry."""

    if (
        type(task) is not Task
        or task.type != task_type
        or task.session_id is not None
        or type(task.input) is not dict
        or set(task.input) != {"dispatch"}
    ):
        return None
    try:
        envelope = _QueuedDispatchEnvelope.model_validate(task.input["dispatch"])
    except (TypeError, ValueError):
        return None
    if not _task_matches_queued_dispatch(
        task,
        task_type=task_type,
        parent_task_id=envelope.request.task_id,
        envelope=envelope,
    ):
        return None
    return envelope


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


def _queued_dispatch_failure_diagnostic(
    runtime: _DurableDispatchRuntime,
    error: BaseException,
    *,
    empty_message: str,
    nonportable_message: str,
) -> ExceptionDiagnostic:
    """Keep private session authority out of durable profile-rejection diagnostics."""

    if isinstance(error, ExecutionProfileMismatchError):
        return ExceptionDiagnostic(
            message="Queued dispatch execution profile did not match its durable requirement.",
            error_type=type(error).__name__,
        )
    return _runtime_exception_diagnostic(
        runtime,
        error,
        empty_message=empty_message,
        nonportable_message=nonportable_message,
    )


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


def _require_profiled_dispatch_runtime(
    runtime: DispatchRuntime,
) -> _ProfiledDispatchRuntime:
    """Reject durable workers without producer/consumer profile authority seams."""

    durable_runtime = _require_dispatch_redaction_boundary(runtime)
    for method_name in (
        "_prepare_queued_dispatch",
        "_dispatch_queued",
        "_queued_dispatch_requests_match",
        "_queued_dispatch_settlement_state",
        "_list_queued_dispatch_terminal_receipts",
        "_acknowledge_queued_dispatch",
    ):
        try:
            method = getattr(durable_runtime, method_name)
        except (AttributeError, TypeError):
            raise TypeError(f"Dispatch runtime {method_name} must be callable.") from None
        if not callable(method):
            raise TypeError(f"Dispatch runtime {method_name} must be callable.")
    return cast("_ProfiledDispatchRuntime", durable_runtime)


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
