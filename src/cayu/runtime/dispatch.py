from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
    exception_suppresses_context,
)
from cayu._exception_groups import set_exception_cause as _set_exception_cause
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._diagnostics import ExceptionDiagnostic
from cayu.runtime._durable_subagents import (
    DurableSubagentSubmissionIntent,
    copy_durable_subagent_submission_intent,
    durable_dispatch_queue_task_id,
    is_durable_subagent_authority_rejected,
    is_durable_subagent_worker_incompatible,
)
from cayu.runtime._durable_worker_loop import (
    DurableWorkerCadence,
    DurableWorkerDemandPolicy,
    DurableWorkerMetrics,
    DurableWorkerPoller,
    DurableWorkerStep,
    record_task_admission_to_claim_latency,
    run_durable_lease_heartbeat,
    run_durable_worker_loop,
    validate_worker_interval,
    wait_or_stop,
)
from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary
from cayu.runtime._task_store_operation_boundary import (
    task_store_cancellation_reconciliation_capability_is_complete,
)
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileIdentity,
    ExecutionProfileMismatchError,
    _ExecutionProfileAdmissionRequestRejected,
    copy_execution_profile_adoption_intent,
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
    ResumeRequest,
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
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskClaimLost,
    TaskCompletionDecisionRequired,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _task_cancellation_requested,
    _task_cancellation_terminalization_request,
    _terminalize_claimed_task_or_detect_peer_winner,
    task_create_with_runtime_invocation,
)
from cayu.runtime.tool_exposure import ToolCapabilityCeiling, copy_tool_capability_ceiling
from cayu.runtime.tool_grants import TargetedToolGrant, validate_targeted_tool_grants
from cayu.runtime.workspace_observation_recovery import (
    is_workspace_observation_recovery_rejected,
)
from cayu.vaults import SecretRedactor

logger = logging.getLogger(__name__)
_DISPATCH_DIAGNOSTIC_MAX_BYTES = 4096
_RUN_WORKER_RECONCILIATION_DISPATCHER: ContextVar[object | None] = ContextVar(
    "cayu_dispatch_worker_reconciliation_dispatcher",
    default=None,
)
_RUN_WORKER_DEMAND_POLLER: ContextVar[tuple[object, DurableWorkerPoller] | None] = ContextVar(
    "cayu_dispatch_worker_demand_poller",
    default=None,
)
_RUN_WORKER_METRICS: ContextVar[DurableWorkerMetrics | None] = ContextVar(
    "cayu_dispatch_worker_metrics",
    default=None,
)
_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)


@dataclass(frozen=True)
class _DispatchOperationDrainOutcome:
    """Failures retained while one cancellation-fenced dispatch becomes quiescent."""

    fence_failure: BaseException | None = None
    operation_failure: BaseException | None = None
    settlement_failure: BaseException | None = None


class _DispatchLeaseAuthority:
    """Serialize renewal and release of one exact queue-task lease."""

    def __init__(self, lease_expires_at: datetime) -> None:
        self.lease_expires_at = lease_expires_at
        self.lock = asyncio.Lock()


def _conservative_dispatch_lease_deadline(started: float, lease_seconds: int) -> float:
    """Retain one-third of the durable lease for a fail-closed drain handoff."""

    return started + (lease_seconds * 2 / 3)


def _consume_dispatch_heartbeat_outcome(task: asyncio.Task[Task]) -> None:
    """Observe a store heartbeat retained past the dispatch owner's deadline."""

    with contextlib.suppress(BaseException):
        task.result()


def _attach_dispatch_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
    *,
    label: str,
) -> None:
    """Retain one secondary failure without replacing the primary graph."""

    if _dispatch_exception_graph_contains_identity(primary, secondary):
        return
    existing = exception_cause(primary)
    if existing is None:
        _set_exception_cause(primary, secondary)
        return
    if _dispatch_exception_graph_contains_identity(existing, secondary):
        return
    _set_exception_cause(primary, BaseExceptionGroup(label, [existing, secondary]))


def _dispatch_exception_graph_contains_identity(
    error: BaseException,
    target: BaseException,
) -> bool:
    """Inspect one runtime-owned dispatch exception graph by identity."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if candidate is target:
            return True
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        elif not exception_suppresses_context(candidate):
            context = exception_context(candidate)
            if context is not None:
                pending.append(context)
    return False


def _dispatch_failure_contains_process_control(error: BaseException) -> bool:
    """Return whether cleanup carried one process-control signal anywhere."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, _PROCESS_CONTROL_SIGNALS):
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        elif not exception_suppresses_context(candidate):
            context = exception_context(candidate)
            if context is not None:
                pending.append(context)
    return False


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
    # None preserves the durable maximum; an explicit subset narrows it permanently.
    tool_capability_ceiling: ToolCapabilityCeiling | None = None
    # Fresh grants apply only to the newly admitted ordinary interaction.
    tool_grants: tuple[TargetedToolGrant, ...] = ()
    profile_adoption: ExecutionProfileAdoptionIntent | None = None
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

    @field_validator("tool_capability_ceiling", mode="before")
    @classmethod
    def copy_tool_capability_ceiling(
        cls,
        value: object,
    ) -> ToolCapabilityCeiling | None:
        if value is None:
            return None
        if isinstance(value, ToolCapabilityCeiling):
            return copy_tool_capability_ceiling(value)
        return ToolCapabilityCeiling.model_validate(value)

    @field_validator("tool_grants", mode="before")
    @classmethod
    def copy_tool_grants(cls, value: object) -> tuple[TargetedToolGrant, ...]:
        return validate_targeted_tool_grants(value)

    @field_validator("profile_adoption", mode="before")
    @classmethod
    def copy_profile_adoption(
        cls,
        value: object,
    ) -> ExecutionProfileAdoptionIntent | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileAdoptionIntent):
            return copy_execution_profile_adoption_intent(value)
        return ExecutionProfileAdoptionIntent.model_validate(value)

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


class _SessionInvocationRuntime(Protocol):
    """Narrow source of authenticated session invocation provenance."""

    async def session_invocation_for_dispatch(
        self,
        session_id: str,
    ) -> SessionInvocationBinding:
        """Load trusted immutable provenance for a durable dispatch target."""


class _PreparedSubagentSubmissionRuntime(_SessionInvocationRuntime, Protocol):
    """Session-side operations needed to settle prepared queue publication."""

    async def _queued_dispatch_settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        """Classify exact session-side evidence before task terminalization."""

    async def _acknowledge_queued_dispatch(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> None:
        """Release terminal-evidence retention after queue terminalization."""


class _DurableDispatchRuntime(DispatchRuntime, _SessionInvocationRuntime, Protocol):
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


_QUEUED_DISPATCH_RECORD_TYPE = "cayu.queued-dispatch"
_QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION = 2
_QUEUED_DISPATCH_INVOCATION_CONTROL_SCHEMA_VERSION = 3
_QUEUED_DISPATCH_EXACT_FORK_SCHEMA_VERSION = 4
_QUEUED_DISPATCH_INVOCATION_CONTROL_FIELDS = (
    "tool_capability_ceiling",
    "tool_grants",
    "profile_adoption",
)


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


@dataclass(frozen=True, slots=True)
class _StalledSessionRecovery:
    """Outcome of one dispatcher-owned incomplete-session recovery attempt."""

    recovered: bool = False
    permanent_rejection: BaseException | None = None

    def __post_init__(self) -> None:
        if type(self.recovered) is not bool:
            raise TypeError("recovered must be a bool.")
        if self.permanent_rejection is not None and not (
            is_workspace_observation_recovery_rejected(self.permanent_rejection)
        ):
            raise TypeError(
                "permanent_rejection must be runtime-owned workspace recovery evidence."
            )
        if self.recovered and self.permanent_rejection is not None:
            raise ValueError("Recovered sessions cannot also retain permanent rejection.")


class _PreparedSubagentAlreadyAdmitted(RuntimeError):
    """Retryable evidence that a prior worker crossed child admission."""


class _QueuedDispatchEnvelope(BaseModel):
    """Runtime-owned authority persisted before queued work becomes claimable."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.queued-dispatch"] = _QUEUED_DISPATCH_RECORD_TYPE
    schema_version: Literal[2, 3, 4] = _QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION
    queue_task_id: str
    dispatch_operation_id: str
    terminal_event_id: str
    request_sha256: str
    session_instance_fingerprint: str
    request: DispatchRequest
    source_profile: ExecutionProfileIdentity
    required_profile: ExecutionProfileIdentity
    operation_kind: Literal["resume", "prepared_subagent"] = "resume"
    prepared_subagent: DurableSubagentSubmissionIntent | None = None
    exact_fork_source_state_sha256: str | None = None

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

    @field_validator("prepared_subagent", mode="before")
    @classmethod
    def copy_prepared_subagent(
        cls,
        value: object,
    ) -> DurableSubagentSubmissionIntent | None:
        if value is None:
            return None
        if type(value) is DurableSubagentSubmissionIntent:
            return copy_durable_subagent_submission_intent(value)
        return DurableSubagentSubmissionIntent.model_validate(value)

    @model_validator(mode="after")
    def validate_authority_tuple(self) -> _QueuedDispatchEnvelope:
        if self.exact_fork_source_state_sha256 is not None and (
            len(self.exact_fork_source_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.exact_fork_source_state_sha256
            )
        ):
            raise ValueError("exact_fork_source_state_sha256 must be a lowercase SHA-256 digest.")
        required_schema_version = _queued_dispatch_schema_version(
            self.request,
            exact_fork_source_state_sha256=self.exact_fork_source_state_sha256,
        )
        if self.schema_version != required_schema_version:
            raise ValueError(
                "Queued dispatch schema version conflicts with its protocol authority."
            )
        if self.operation_kind == "resume":
            if self.prepared_subagent is not None:
                raise ValueError("Resume dispatch cannot carry prepared-subagent authority.")
        else:
            intent = self.prepared_subagent
            if intent is None:
                raise ValueError("Prepared-subagent dispatch requires submission authority.")
            if self.exact_fork_source_state_sha256 is not None:
                raise ValueError(
                    "Prepared-subagent dispatch cannot carry exact-fork resume authority."
                )
            if (
                self.request != _prepared_subagent_dispatch_request(intent)
                or self.source_profile != intent.child_execution_profile
                or self.required_profile != intent.child_execution_profile
            ):
                raise ValueError(
                    "Prepared-subagent dispatch conflicts with its submission authority."
                )
        request_sha256 = _queued_dispatch_request_sha256(
            self.request,
            schema_version=self.schema_version,
        )
        if self.request_sha256 != request_sha256:
            raise ValueError("Queued dispatch request digest does not match its request.")
        operation_id = _queued_dispatch_operation_id(
            queue_task_id=self.queue_task_id,
            request=self.request,
            request_sha256=request_sha256,
            session_instance_fingerprint=self.session_instance_fingerprint,
            source_profile=self.source_profile,
            required_profile=self.required_profile,
            schema_version=self.schema_version,
            exact_fork_source_state_sha256=self.exact_fork_source_state_sha256,
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
LEGACY_PREPARED_SUBAGENT_DISPATCH_TASK_TYPE_SUFFIX = ".prepared-subagent.v1"
PREPARED_SUBAGENT_DISPATCH_TASK_TYPE_SUFFIX = ".prepared-subagent.v2"
INVOCATION_CONTROL_DISPATCH_TASK_TYPE_SUFFIX = ".invocation-controls.v3"
EXACT_FORK_DISPATCH_TASK_TYPE_SUFFIX = ".exact-fork.v4"
DISPATCH_CONFLICT_RECOVERY_REASON = "dispatch_conflict_worker_crash_recovery"

_STALLED_RECOVERED_ACTIONS = {
    IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
    IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION,
    IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION,
    IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,
    IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT,
    IncompleteSessionRecoveryAction.PENDING_APPROVAL,
    IncompleteSessionRecoveryAction.PENDING_USER_INPUT,
    IncompleteSessionRecoveryAction.AMBIGUOUS_PENDING_USER_INPUT,
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
        if self._task_type.endswith(
            (
                LEGACY_PREPARED_SUBAGENT_DISPATCH_TASK_TYPE_SUFFIX,
                PREPARED_SUBAGENT_DISPATCH_TASK_TYPE_SUFFIX,
                INVOCATION_CONTROL_DISPATCH_TASK_TYPE_SUFFIX,
                EXACT_FORK_DISPATCH_TASK_TYPE_SUFFIX,
            )
        ):
            raise ValueError("task_type uses a reserved dispatch protocol suffix.")
        self._prepared_subagent_task_type = require_clean_nonblank(
            self._task_type + PREPARED_SUBAGENT_DISPATCH_TASK_TYPE_SUFFIX,
            "prepared_subagent_task_type",
        )
        self._invocation_control_task_type = require_clean_nonblank(
            self._task_type + INVOCATION_CONTROL_DISPATCH_TASK_TYPE_SUFFIX,
            "invocation_control_task_type",
        )
        self._exact_fork_task_type = require_clean_nonblank(
            self._task_type + EXACT_FORK_DISPATCH_TASK_TYPE_SUFFIX,
            "exact_fork_task_type",
        )
        self._next_claim_task_type_index = 0
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
        self._terminal_receipt_reconciliation_task_generation: int | None = None
        self._terminal_receipt_reconciliation_generation = 0
        self._terminal_receipt_reconciliation_settled_count = 0
        self._startup_terminal_receipt_reconciliation_pending = True

    @property
    def task_type(self) -> str:
        """Return the durable queue namespace used by this dispatcher."""

        return self._task_type

    @property
    def prepared_subagent_task_type(self) -> str:
        """Return the versioned queue namespace reserved for prepared children."""

        return self._prepared_subagent_task_type

    @property
    def invocation_control_task_type(self) -> str:
        """Return the versioned namespace for expanded resume controls."""

        return self._invocation_control_task_type

    @property
    def exact_fork_task_type(self) -> str:
        """Return the versioned namespace for exact-fork child dispatches."""

        return self._exact_fork_task_type

    @property
    def task_store(self) -> TaskStore:
        """Return the exact task store that owns this dispatcher's leases."""

        return self._tasks

    def _claim_task_types(self) -> tuple[str, ...]:
        """Return independently claimable protocol namespaces in fair-poll order."""

        return (
            self._prepared_subagent_task_type,
            self._exact_fork_task_type,
            self._invocation_control_task_type,
            self._task_type,
        )

    def _task_type_for_envelope(self, envelope: _QueuedDispatchEnvelope) -> str:
        """Bind each envelope protocol to the only namespace allowed to carry it."""

        if type(envelope) is not _QueuedDispatchEnvelope:
            raise TypeError("Queued dispatch requires an exact envelope.")
        if envelope.operation_kind == "resume":
            if envelope.schema_version == _QUEUED_DISPATCH_EXACT_FORK_SCHEMA_VERSION:
                return self._exact_fork_task_type
            if envelope.schema_version == _QUEUED_DISPATCH_INVOCATION_CONTROL_SCHEMA_VERSION:
                return self._invocation_control_task_type
            return self._task_type
        intent = envelope.prepared_subagent
        if intent is None or intent.queue_task_type != self._prepared_subagent_task_type:
            raise ValueError("Prepared subagent dispatch uses an unsupported queue namespace.")
        return self._prepared_subagent_task_type

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
            task_type = existing.type
            envelope = _existing_queued_dispatch_envelope(
                existing,
                task_type=task_type,
            )
            if (
                task_type not in self._claim_task_types()
                or envelope is None
                or self._task_type_for_envelope(envelope) != task_type
                or not await durable_runtime._queued_dispatch_requests_match(
                    envelope.request,
                    request,
                )
            ):
                raise RuntimeError("Existing task conflicts with the queued dispatch authority.")
            settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
            session_binding = await _load_dispatch_session_invocation(
                durable_runtime,
                _queued_dispatch_authority_session_id(envelope),
            )
            _require_dispatch_task_authority(
                existing,
                envelope=envelope,
                session_binding=session_binding,
                task_type=task_type,
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
        task_type = self._task_type_for_envelope(envelope)
        request = envelope.request
        # The queue task must be session-unbound (``session_id is None``) to be claimable by
        # a worker pool; the target session_id rides inside the serialized request payload.
        create_request = task_create_with_runtime_invocation(
            TaskCreate(
                task_id=queue_task_id,
                type=task_type,
                parent_task_id=request.task_id,
                input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
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
                    task_type=task_type,
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
            task_type=task_type,
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
            envelope=envelope,
            session_binding=session_binding,
            task_type=task_type,
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

    async def _submit_prepared_subagent(
        self,
        runtime: _PreparedSubagentSubmissionRuntime,
        envelope: _QueuedDispatchEnvelope,
    ) -> DispatchHandle:
        """Publish one pre-created child through the existing durable task queue."""

        durable_runtime = _require_prepared_subagent_submission_runtime(runtime)
        envelope = _copy_queued_dispatch_envelope(envelope)
        intent = envelope.prepared_subagent
        if envelope.operation_kind != "prepared_subagent" or intent is None:
            raise ValueError("Prepared subagent submission requires exact queue authority.")
        prepared_task_type = self._prepared_subagent_task_type
        if intent.queue_task_type != prepared_task_type:
            raise ValueError("Durable subagent queue task type conflicts with its dispatcher.")
        parent_binding = await _load_dispatch_session_invocation(
            durable_runtime,
            intent.parent_session_id,
        )
        existing = await self._tasks.load_task(intent.queue_task_id)
        idempotent_submission = existing is not None
        if existing is None:
            create_request = task_create_with_runtime_invocation(
                TaskCreate(
                    task_id=intent.queue_task_id,
                    type=prepared_task_type,
                    parent_task_id=intent.parent_task_id,
                    input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
                ),
                source=TaskExecutionSource.TASK_DISPATCH,
                session_invocation=parent_binding,
            )
            try:
                existing = await self._tasks.create_task(create_request)
            except Exception as publication_failure:
                try:
                    existing = await self._tasks.load_task(intent.queue_task_id)
                except Exception as reconciliation_failure:
                    publication_failure.add_note(
                        "Durable subagent task publication reconciliation also failed: "
                        f"{type(reconciliation_failure).__name__}."
                    )
                    raise publication_failure from reconciliation_failure
                if existing is None:
                    raise
                idempotent_submission = True
        existing_envelope = _existing_queued_dispatch_envelope(
            existing,
            task_type=prepared_task_type,
        )
        if existing_envelope != envelope:
            raise RuntimeError("Existing task conflicts with durable subagent authority.")
        _require_dispatch_task_authority(
            existing,
            envelope=envelope,
            session_binding=parent_binding,
            task_type=prepared_task_type,
        )
        if existing.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            settlement = await durable_runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
            if settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE:
                await self._acknowledge_terminal_task(durable_runtime, existing, envelope)
        return self._handle(
            envelope.request,
            DispatchStatus.SUBMITTED,
            queue_task_id=existing.id,
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
        malformed (in which case the task is failed before returning). Direct callers
        reconcile pending terminal receipts by default. ``run_worker`` suppresses that
        duplicate entrance in its task-local context because its independently elected
        periodic role owns that cadence.
        """
        durable_runtime = _require_profiled_dispatch_runtime(runtime)
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        if not task_store_cancellation_reconciliation_capability_is_complete(self._tasks):
            raise NotImplementedError(
                "Queued dispatch workers require complete idempotent ordinary-task "
                "cancellation reconciliation before they can claim work."
            )
        if (
            _RUN_WORKER_RECONCILIATION_DISPATCHER.get() is not self
            and self._startup_terminal_receipt_reconciliation_pending
        ):
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
        claim_task_types = self._claim_task_types()
        loop = asyncio.get_running_loop()
        poller_context = _RUN_WORKER_DEMAND_POLLER.get()
        demand_poller = (
            poller_context[1] if poller_context is not None and poller_context[0] is self else None
        )
        claim_type_limit = (
            len(claim_task_types) if demand_poller is None or demand_poller.has_pending_hint else 1
        )

        async def claim_next_task() -> tuple[Task, str, _DispatchLeaseAuthority, float] | None:
            claim_start_index = (
                self._next_claim_task_type_index
                if demand_poller is None
                else demand_poller.reserve_claim_query_index(len(claim_task_types))
            )
            for offset in range(claim_type_limit):
                candidate_index = (claim_start_index + offset) % len(claim_task_types)
                candidate_task_type = claim_task_types[candidate_index]
                claim_started_monotonic = loop.time()
                candidate = await self._tasks.claim_task(
                    worker_id,
                    # Each namespace remains FIFO. Alternate successful claims so a
                    # continuously busy queue cannot starve the other protocol.
                    TaskQuery(type=candidate_task_type, order_by=TaskOrder.CREATED_AT_ASC),
                    lease_seconds=self._lease_seconds,
                )
                if candidate is None:
                    if demand_poller is None and claim_type_limit == 1:
                        self._next_claim_task_type_index = (candidate_index + 1) % len(
                            claim_task_types
                        )
                    elif demand_poller is not None and offset + 1 < claim_type_limit:
                        # The poller records the final empty or successful store call.
                        # Preserve each earlier authoritative operation in a hinted
                        # full-union audit so metrics remain operation-exact.
                        demand_poller.metrics.claim_completed(claimed=False)
                    continue
                if candidate.lease_expires_at is None:  # pragma: no cover - claim contract
                    raise TaskClaimLost("Queued dispatch claim has no worker lease.")
                lease_authority = _DispatchLeaseAuthority(candidate.lease_expires_at)
                claim_deadline_monotonic = _conservative_dispatch_lease_deadline(
                    claim_started_monotonic,
                    self._lease_seconds,
                )
                if loop.time() >= claim_deadline_monotonic:
                    with contextlib.suppress(TaskClaimLost):
                        await self._release_claimed_task(
                            candidate.id,
                            worker_id,
                            lease_authority,
                        )
                    return None
                if demand_poller is None:
                    self._next_claim_task_type_index = (candidate_index + 1) % len(claim_task_types)
                return (
                    candidate,
                    candidate_task_type,
                    lease_authority,
                    claim_deadline_monotonic,
                )
            return None

        if demand_poller is not None:
            claim = await demand_poller.claim(
                claim_next_task,
                maximum_active_s=self._lease_seconds,
            )
            claimed = claim.value
        else:
            claimed = await claim_next_task()
        if claimed is None:
            return None
        if demand_poller is not None:
            demand_poller.metrics.handler_started()
        task, claimed_task_type, lease_authority, claim_deadline_monotonic = claimed
        if demand_poller is not None:
            record_task_admission_to_claim_latency(
                demand_poller.metrics,
                admitted_at=task.created_at,
                claimed_at=datetime.now(UTC),
            )
        # Fail malformed or unauthenticated queue authority terminally rather than letting
        # the task be reclaimed and re-run forever. Only the immutable task row is consulted
        # in this phase: store-backed session authority remains operational and retryable.
        payload = task.input.get("dispatch")
        try:
            if type(payload) is not dict:
                raise ValueError("dispatch task envelope payload is not an object")
            envelope = _QueuedDispatchEnvelope.model_validate(payload)
            if self._task_type_for_envelope(envelope) != claimed_task_type:
                raise ValueError("dispatch task protocol conflicts with its queue namespace")
            if not _claimed_task_matches_queued_dispatch(
                task,
                task_type=claimed_task_type,
                worker_id=worker_id,
                envelope=envelope,
            ):
                raise ValueError("dispatch task row conflicts with its envelope")
        except (TypeError, ValueError) as exc:
            await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                lease_authority=lease_authority,
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
                lease_authority=lease_authority,
                error=exc,
                envelope=envelope,
            )
        except Exception as exc:
            await self._release_claimed_dispatch_after_failure(
                task=task,
                worker_id=worker_id,
                lease_authority=lease_authority,
                failure=exc,
            )
            raise

        try:
            session_binding = await _load_dispatch_session_invocation(
                durable_runtime,
                _queued_dispatch_authority_session_id(envelope),
            )
        except _QueuedDispatchAuthorityRejected as exc:
            return await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                lease_authority=lease_authority,
                error=exc,
                envelope=envelope,
            )
        except Exception as exc:
            await self._release_claimed_dispatch_after_failure(
                task=task,
                worker_id=worker_id,
                lease_authority=lease_authority,
                failure=exc,
            )
            raise
        try:
            _require_dispatch_task_authority(
                task,
                envelope=envelope,
                session_binding=session_binding,
                task_type=claimed_task_type,
            )
        except (TypeError, ValueError) as exc:
            return await self._reject_claimed_dispatch(
                durable_runtime,
                task=task,
                worker_id=worker_id,
                lease_authority=lease_authority,
                error=exc,
                envelope=envelope,
            )

        if settlement.state is not _QueuedDispatchSettlementState.NOT_ADMITTED:
            status = settlement.terminal_status or DispatchStatus.SUBMITTED
            if loop.time() >= claim_deadline_monotonic:
                return await self._reclaimed_dispatch_handle(task, envelope)
            heartbeat_stop, heartbeat = self._start_heartbeat(
                task.id,
                worker_id,
                durable_runtime,
                claim_deadline_monotonic=claim_deadline_monotonic,
                lease_authority=lease_authority,
            )
            try:
                try:
                    return await self._await_with_task_lease(
                        self._terminalize(
                            durable_runtime,
                            task,
                            worker_id,
                            request,
                            status,
                            {"status": status.value, **_queued_dispatch_evidence(envelope)},
                            envelope=envelope,
                            lease_authority=lease_authority,
                            settlement=settlement,
                        ),
                        heartbeat,
                        task=task,
                        worker_id=worker_id,
                        dispatcher=self,
                        lease_authority=lease_authority,
                    )
                except TaskClaimLost:
                    return await self._reclaimed_dispatch_handle(task, envelope)
            finally:
                await self._stop_heartbeat(heartbeat_stop, heartbeat)

        # Heartbeat in the background so the lease survives long gaps between events (a slow
        # model/tool turn would otherwise let the lease lapse and another worker re-run it).
        # The outer try/finally keeps the heartbeat alive THROUGH terminalization — a slow
        # complete/fail/release must not let the lease expire and get the task reclaimed and
        # run a second time — and always stops it, including on CancelledError (graceful
        # worker shutdown), which neither except below catches.
        status = DispatchStatus.SUBMITTED
        if loop.time() >= claim_deadline_monotonic:
            return await self._reclaimed_dispatch_handle(task, envelope)
        if task.lease_expires_at is None:  # pragma: no cover - claim contract
            raise TaskClaimLost("Queued dispatch claim has no worker lease.")
        task = await self._tasks.mark_claimed_task_execution_started(
            task.id,
            worker_id,
            task.lease_expires_at,
        )
        if loop.time() >= claim_deadline_monotonic:
            requested = await self._request_dispatch_cancellation_fence(
                task,
                worker_id,
                {"code": "dispatch_execution_start_acknowledgement_consumed_lease"},
                lease_authority=lease_authority,
            )
            if requested is not None and _task_cancellation_requested(requested):
                await self._settle_dispatch_cancellation_after_quiescence(
                    task.id,
                    worker_id,
                )
            return self._handle(
                request,
                DispatchStatus.CANCELLED,
                queue_task_id=task.id,
                envelope=envelope,
            )
        heartbeat_stop, heartbeat = self._start_heartbeat(
            task.id,
            worker_id,
            durable_runtime,
            claim_deadline_monotonic=claim_deadline_monotonic,
            lease_authority=lease_authority,
        )
        try:
            try:

                async def dispatch_owned() -> DispatchStatus:
                    owned_status = status
                    async for event in durable_runtime._dispatch_queued(envelope):
                        owned_status = _dispatch_status_after_event(
                            event,
                            fallback=owned_status,
                        )
                    return owned_status

                status = await self._await_with_task_lease(
                    dispatch_owned(),
                    heartbeat,
                    task=task,
                    worker_id=worker_id,
                    dispatcher=self,
                    lease_authority=lease_authority,
                )
            except TaskClaimLost:
                return await self._reclaimed_dispatch_handle(task, envelope)
            except (SessionRunFenced, SessionStatusConflict):
                # The session is already being run by another worker — requeue rather than
                # fail, so it runs once that session frees up (per-session serialization).
                # The same rule applies while terminal hooks or trailing cleanup retain the
                # prior invocation's profile fence. After a worker crash, recover stale
                # session ownership so the requeued dispatch can proceed.
                recovery = await self._recover_stalled_session(
                    durable_runtime,
                    request,
                )
                if recovery.permanent_rejection is not None:
                    return await self._reject_claimed_dispatch(
                        durable_runtime,
                        task=task,
                        worker_id=worker_id,
                        lease_authority=lease_authority,
                        error=recovery.permanent_rejection,
                        envelope=envelope,
                    )
                try:
                    await self._release_claimed_task(
                        task.id,
                        worker_id,
                        lease_authority,
                    )
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
                        recovered_session=recovery.recovered,
                    )
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task.id,
                    envelope=envelope,
                    requeued=True,
                    recovered_session=recovery.recovered,
                )
            except (
                ExecutionProfileMismatchError,
                _QueuedDispatchAuthorityRejected,
                TaskCompletionDecisionRequired,
            ) as exc:
                return await self._reject_claimed_dispatch(
                    durable_runtime,
                    task=task,
                    worker_id=worker_id,
                    lease_authority=lease_authority,
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
                        lease_authority=lease_authority,
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
                        lease_authority=lease_authority,
                    )
                    raise combined_failure from None
                if settlement.state is _QueuedDispatchSettlementState.NOT_ADMITTED:
                    if is_durable_subagent_authority_rejected(exc):
                        return await self._reject_claimed_dispatch(
                            durable_runtime,
                            task=task,
                            worker_id=worker_id,
                            lease_authority=lease_authority,
                            error=exc,
                            envelope=envelope,
                        )
                    if is_durable_subagent_worker_incompatible(exc):
                        try:
                            await self._release_claimed_task(
                                task.id,
                                worker_id,
                                lease_authority,
                            )
                        except TaskClaimLost:
                            logger.warning(
                                "dispatch %s lost its lease while requeueing an "
                                "incompatible prepared-subagent worker",
                                request.dispatch_id,
                            )
                            return self._handle(
                                request,
                                DispatchStatus.SUBMITTED,
                                queue_task_id=task.id,
                                envelope=envelope,
                                reclaimed=True,
                            )
                        except asyncio.CancelledError as cancellation:
                            raise cancellation from exc
                        except Exception as release_error:
                            raise ExceptionGroup(
                                "Prepared-subagent worker incompatibility and task "
                                "release both failed.",
                                [exc, release_error],
                            ) from None
                        return self._handle(
                            request,
                            DispatchStatus.SUBMITTED,
                            queue_task_id=task.id,
                            envelope=envelope,
                            requeued=True,
                        )
                    if isinstance(
                        exc,
                        _ExecutionProfileAdmissionRequestRejected,
                    ):
                        return await self._reject_claimed_dispatch(
                            durable_runtime,
                            task=task,
                            worker_id=worker_id,
                            lease_authority=lease_authority,
                            error=exc,
                            envelope=envelope,
                        )
                    await self._release_claimed_dispatch_after_failure(
                        task=task,
                        worker_id=worker_id,
                        failure=exc,
                        lease_authority=lease_authority,
                    )
                    raise
                try:
                    return await self._await_with_task_lease(
                        self._terminalize(
                            durable_runtime,
                            task,
                            worker_id,
                            request,
                            DispatchStatus.FAILED,
                            {
                                **failure_payload,
                                **_queued_dispatch_evidence(envelope),
                                "status": DispatchStatus.FAILED.value,
                            },
                            envelope=envelope,
                            lease_authority=lease_authority,
                            settlement=settlement,
                        ),
                        heartbeat,
                        task=task,
                        worker_id=worker_id,
                        dispatcher=self,
                        lease_authority=lease_authority,
                    )
                except TaskClaimLost:
                    return await self._reclaimed_dispatch_handle(task, envelope)
            # A run can fail in-band (a SESSION_FAILED event, not an exception); record that as
            # a failed task so failure queries and retries see it, not a COMPLETED one.
            try:
                return await self._await_with_task_lease(
                    self._terminalize(
                        durable_runtime,
                        task,
                        worker_id,
                        request,
                        status,
                        {"status": status.value, **_queued_dispatch_evidence(envelope)},
                        envelope=envelope,
                        lease_authority=lease_authority,
                    ),
                    heartbeat,
                    task=task,
                    worker_id=worker_id,
                    dispatcher=self,
                    lease_authority=lease_authority,
                )
            except TaskClaimLost:
                return await self._reclaimed_dispatch_handle(task, envelope)
        finally:
            await self._stop_heartbeat(heartbeat_stop, heartbeat)

    async def _reject_claimed_dispatch(
        self,
        runtime: _DurableDispatchRuntime,
        *,
        task: Task,
        worker_id: str,
        lease_authority: _DispatchLeaseAuthority,
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
            async with lease_authority.lock:
                peer_terminalization_won = await self._commit_task_terminal(
                    task_id=task.id,
                    worker_id=worker_id,
                    lease_expires_at=lease_authority.lease_expires_at,
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
                    task_type=self._task_type_for_envelope(envelope),
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
        lease_authority: _DispatchLeaseAuthority,
        failure: Exception,
    ) -> None:
        """Make pre-admission work reclaimable without hiding a release failure."""

        try:
            await self._release_claimed_task(
                task.id,
                worker_id,
                lease_authority,
            )
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

    async def _release_claimed_task(
        self,
        task_id: str,
        worker_id: str,
        lease_authority: _DispatchLeaseAuthority,
    ) -> Task:
        """Release only the exact lease generation acknowledged by this worker."""

        async with lease_authority.lock:
            return await self._tasks.release_task(
                task_id,
                worker_id,
                lease_expires_at=lease_authority.lease_expires_at,
            )

    async def _terminalize(
        self,
        runtime: _ProfiledDispatchRuntime,
        task: Task,
        worker_id: str,
        request: DispatchRequest,
        status: DispatchStatus,
        payload: dict[str, Any],
        *,
        envelope: _QueuedDispatchEnvelope,
        lease_authority: _DispatchLeaseAuthority,
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
            recovery = await self._recover_stalled_session(runtime, request)
            if recovery.permanent_rejection is not None:
                rejected = await self._reject_claimed_dispatch(
                    runtime,
                    task=task,
                    worker_id=worker_id,
                    lease_authority=lease_authority,
                    error=recovery.permanent_rejection,
                    envelope=envelope,
                )
                if rejected is None:
                    raise AssertionError("Authenticated dispatch rejection lost its envelope.")
                return rejected
            try:
                await self._release_claimed_task(
                    task.id,
                    worker_id,
                    lease_authority,
                )
            except TaskClaimLost:
                logger.warning(
                    "dispatch %s lost its lease while retaining incomplete terminal evidence",
                    request.dispatch_id,
                )
                return self._handle(
                    request,
                    DispatchStatus.SUBMITTED,
                    queue_task_id=task.id,
                    envelope=envelope,
                    reclaimed=True,
                    recovered_session=recovery.recovered,
                )
            return self._handle(
                request,
                DispatchStatus.SUBMITTED,
                queue_task_id=task.id,
                envelope=envelope,
                requeued=True,
                recovered_session=recovery.recovered,
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
            async with lease_authority.lock:
                peer_terminalization_won = await self._commit_task_terminal(
                    task_id=task.id,
                    worker_id=worker_id,
                    lease_expires_at=lease_authority.lease_expires_at,
                    kind=kind,
                    payload=payload,
                )
            terminal_task = await self._tasks.load_task(task.id)
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
                    queue_task_id=task.id,
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
                queue_task_id=task.id,
                envelope=envelope,
                reclaimed=True,
            )
        return self._handle(
            request,
            authoritative_status,
            queue_task_id=task.id,
            envelope=envelope,
        )

    async def _commit_task_terminal(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_expires_at: datetime | None,
        kind: TaskTerminalKind,
        payload: dict[str, Any],
    ) -> bool:
        return await _terminalize_claimed_task_or_detect_peer_winner(
            self._tasks,
            TaskTerminalizationRequest(
                task_id=task_id,
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
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
    ) -> _StalledSessionRecovery:
        """Best-effort finalization of a session stranded in a live status by a crashed worker.

        Uses the runtime's incomplete-session recovery when available while the
        durable redaction capabilities remain mandatory. The store atomically checks the
        durable activity horizon and increments the run epoch before recovery, so a
        genuinely live run is left alone and an evicted worker cannot write after the
        decision. Permanent runtime-authenticated observation corruption is returned
        separately so the queue task can settle instead of being reclaimed forever.
        """
        recover = getattr(runtime, "recover_incomplete_session", None)
        if recover is None:
            return _StalledSessionRecovery()
        try:
            result = await recover(
                IncompleteSessionRecoveryRequest(
                    session_id=request.session_id,
                    inactive_for_seconds=self._recover_stalled_after_seconds,
                    reason=DISPATCH_CONFLICT_RECOVERY_REASON,
                    metadata={"dispatch_id": request.dispatch_id},
                )
            )
        except Exception as exc:
            permanent_rejection = exc if is_workspace_observation_recovery_rejected(exc) else None
            logger.warning(
                "dispatch %s could not recover stalled session %s: error_type=%s error=%s",
                request.dispatch_id,
                request.session_id,
                type(exc).__name__,
                _safe_runtime_text(runtime, str(exc)),
            )
            return _StalledSessionRecovery(permanent_rejection=permanent_rejection)
        return _StalledSessionRecovery(
            recovered=bool(_STALLED_RECOVERED_ACTIONS & set(result.actions))
        )

    async def _heartbeat(
        self,
        task_id: str,
        worker_id: str,
        runtime: _DurableDispatchRuntime,
        stop: asyncio.Event,
        *,
        claim_deadline_monotonic: float,
        lease_authority: _DispatchLeaseAuthority,
    ) -> None:
        """Extend the lease while enforcing the last acknowledged local deadline."""
        loop = asyncio.get_running_loop()

        def current_deadline() -> float:
            return claim_deadline_monotonic

        def expired_lease() -> BaseException:
            return TaskClaimLost("Dispatch task heartbeat acknowledgement expired before renewal.")

        async def heartbeat() -> Task:
            nonlocal claim_deadline_monotonic
            async with lease_authority.lock:
                renewal_started = loop.time()
                renewal_task = asyncio.create_task(
                    self._tasks.heartbeat(
                        task_id,
                        worker_id,
                        lease_expires_at=lease_authority.lease_expires_at,
                        extend_seconds=self._lease_seconds,
                    )
                )
                try:
                    completed, _pending = await asyncio.wait(
                        {renewal_task},
                        timeout=max(0.0, claim_deadline_monotonic - loop.time()),
                    )
                except asyncio.CancelledError:
                    renewal_task.add_done_callback(_consume_dispatch_heartbeat_outcome)
                    raise
                if renewal_task not in completed:
                    renewal_task.add_done_callback(_consume_dispatch_heartbeat_outcome)
                    raise TaskClaimLost(
                        "Dispatch task heartbeat acknowledgement did not arrive before "
                        "the last positively known lease deadline."
                    )
                renewed = renewal_task.result()
                if renewed.lease_expires_at is None:  # pragma: no cover - store contract
                    raise TaskClaimLost("Dispatch heartbeat returned no worker lease.")
                lease_authority.lease_expires_at = renewed.lease_expires_at
                claim_deadline_monotonic = _conservative_dispatch_lease_deadline(
                    renewal_started,
                    self._lease_seconds,
                )
                if loop.time() >= claim_deadline_monotonic:
                    raise TaskClaimLost(
                        "Dispatch task heartbeat acknowledgement consumed the renewed lease."
                    )
                return renewed

        async def reconcile_heartbeat_failure(exc: Exception) -> None:
            if isinstance(exc, TaskClaimLost):
                raise exc
            logger.warning(
                "dispatch heartbeat failed for task %s: error_type=%s error=%s",
                task_id,
                type(exc).__name__,
                _safe_runtime_text(runtime, str(exc)),
            )
            return None

        await run_durable_lease_heartbeat(
            heartbeat,
            lease_seconds=self._lease_seconds,
            stop=stop,
            stopped_outcome=None,
            maximum_interval_s=1.0,
            on_failure=reconcile_heartbeat_failure,
            lease_deadline=current_deadline,
            deadline_failure=expired_lease,
            clock=loop.time,
        )

    def _start_heartbeat(
        self,
        task_id: str,
        worker_id: str,
        runtime: _DurableDispatchRuntime,
        *,
        claim_deadline_monotonic: float,
        lease_authority: _DispatchLeaseAuthority,
    ) -> tuple[asyncio.Event, asyncio.Task[None]]:
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(
                task_id,
                worker_id,
                runtime,
                stop,
                claim_deadline_monotonic=claim_deadline_monotonic,
                lease_authority=lease_authority,
            )
        )
        return stop, heartbeat

    async def _request_dispatch_cancellation_fence(
        self,
        task: Task,
        worker_id: str,
        error: dict[str, Any],
        *,
        lease_authority: _DispatchLeaseAuthority,
    ) -> Task | None:
        """Persist a reclaim-blocking marker before draining an opaque dispatch."""

        async with lease_authority.lock:
            while True:
                expected_lease_expires_at = lease_authority.lease_expires_at
                current = await self._tasks.load_task(task.id)
                if current is None:
                    raise TaskClaimLost("Dispatch cancellation fence target no longer exists.")
                if current.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                } or _task_cancellation_requested(current):
                    return current
                if (
                    current.worker_id != worker_id
                    or current.lease_expires_at != expected_lease_expires_at
                ):
                    raise TaskClaimLost(
                        "Dispatch cancellation fence no longer owns the expected lease generation."
                    )
                try:
                    requested = await self._tasks.request_claimed_task_cancellation(
                        task.id,
                        worker_id,
                        expected_lease_expires_at,
                        error,
                    )
                except (GeneratorExit, KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as request_failure:
                    if _dispatch_failure_contains_process_control(request_failure):
                        raise
                    try:
                        current = await self._tasks.load_task(task.id)
                    except (GeneratorExit, KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as reconciliation_failure:
                        if _dispatch_failure_contains_process_control(reconciliation_failure):
                            raise
                        await asyncio.sleep(0.05)
                        continue
                    if current is None:
                        raise request_failure
                    if current.status in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    } or _task_cancellation_requested(current):
                        return current
                    if (
                        current.worker_id != worker_id
                        or current.lease_expires_at != expected_lease_expires_at
                    ):
                        raise TaskClaimLost(
                            "Dispatch cancellation fence lost its exact worker lease."
                        ) from request_failure
                    if isinstance(request_failure, TaskClaimLost):
                        raise
                    await asyncio.sleep(0.05)
                    continue
                if requested.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                } or _task_cancellation_requested(requested):
                    return requested
                await asyncio.sleep(0.05)

    async def _drain_dispatch_operation_under_cancellation_fence(
        self,
        operation_task: asyncio.Task[Any],
        *,
        task: Task,
        worker_id: str,
        lease_authority: _DispatchLeaseAuthority,
        error: dict[str, Any],
    ) -> _DispatchOperationDrainOutcome:
        """Fence one dispatch, retain it to natural settlement, then publish quiescence."""

        requested: Task | None = None
        fence_failure: BaseException | None = None
        try:
            requested = await self._request_dispatch_cancellation_fence(
                task,
                worker_id,
                error,
                lease_authority=lease_authority,
            )
        except BaseException as exc:
            fence_failure = (
                unexpected_child_cancellation_error(
                    exc,
                    operation="Queued dispatch cancellation-fence publication",
                )
                if isinstance(exc, asyncio.CancelledError)
                else exc
            )

        operation_outcome = await await_shielded_task_outcome(operation_task)
        operation_failure = operation_outcome.error
        if isinstance(operation_failure, asyncio.CancelledError):
            operation_failure = unexpected_child_cancellation_error(
                operation_failure,
                operation="Queued dispatch cancellation-fence drain",
            )

        settlement_failure: BaseException | None = None
        if requested is not None and _task_cancellation_requested(requested):
            try:
                await self._settle_dispatch_cancellation_after_quiescence(
                    task.id,
                    worker_id,
                )
            except BaseException as exc:
                settlement_failure = (
                    unexpected_child_cancellation_error(
                        exc,
                        operation="Queued dispatch cancellation-fence settlement",
                    )
                    if isinstance(exc, asyncio.CancelledError)
                    else exc
                )

        return _DispatchOperationDrainOutcome(
            fence_failure=fence_failure,
            operation_failure=operation_failure,
            settlement_failure=settlement_failure,
        )

    async def _settle_dispatch_cancellation_after_quiescence(
        self,
        task_id: str,
        worker_id: str,
    ) -> None:
        """Commit exact quiescence after a lease-loss dispatch has naturally ended."""

        current = await self._tasks.load_task(task_id)
        if current is None or current.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return
        terminalization = _task_cancellation_terminalization_request(
            current,
            worker_id=worker_id,
        )
        if terminalization is not None:
            with contextlib.suppress(TaskClaimLost, TaskTerminalizationConflict):
                await _terminalize_claimed_task_or_detect_peer_winner(
                    self._tasks,
                    terminalization,
                )
        current = await self._tasks.load_task(task_id)
        if current is None or current.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return
        payload = current.status_payload
        if (
            current.worker_id != worker_id
            or current.lease_expires_at is None
            or not _task_cancellation_requested(current)
            or type(payload) is not dict
            or type(payload.get("event")) is not dict
        ):
            raise TaskTerminalizationConflict(
                "Dispatch quiescence lost its durable cancellation authority."
            )
        event = TaskCancellationReconciliationEvent.model_validate(payload["event"])
        cancellation_key = (
            terminalization.idempotency_key
            if terminalization is not None
            else f"task-cancel:v1:{event.id}"
        )
        execution_profile_fingerprint = current.metadata.get("execution_profile_fingerprint")
        effect_fingerprint = current.metadata.get("effect_fingerprint")
        evidence_identity = canonical_durable_json_bytes(
            {
                "schema": "cayu.dispatch-quiescence.v1",
                "task_id": current.id,
                "worker_id": worker_id,
                "cancellation_idempotency_key": cancellation_key,
            },
            "dispatch_quiescence",
        )
        evidence_sha256 = sha256(evidence_identity).hexdigest()
        # The cancellation event and lease expiry are both store-generated.
        # A worker clock skewed ahead of the task store must not make an exact
        # owner-lost reconciliation look as though it came from the future.
        validated_at = max(event.occurred_at, current.lease_expires_at)
        await self._tasks.reconcile_task_cancellation(
            TaskCancellationReconciliationRequest(
                task_id=current.id,
                original_worker_id=worker_id,
                original_lease_expires_at=current.lease_expires_at,
                cancellation_requested_at=event.occurred_at,
                cancellation_idempotency_key=cancellation_key,
                reconciliation_idempotency_key=(f"dispatch-quiescence:v1:{evidence_sha256}"),
                reconciliation_requested_at=validated_at,
                reconciled_by=ResolutionActor(
                    subject="cayu:dispatch-quiescence",
                    source=ResolutionActorSource.SYSTEM,
                ),
                evidence=TaskCancellationReconciliationEvidence(
                    outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                    validator_id="cayu.dispatch",
                    validator_version="1",
                    evidence_id=f"dispatch-quiescence:{evidence_sha256}",
                    evidence_sha256=evidence_sha256,
                    validated_at=validated_at,
                    execution_profile_fingerprint=execution_profile_fingerprint,
                    effect_fingerprint=effect_fingerprint,
                ),
                expected_execution_profile_fingerprint=execution_profile_fingerprint,
                expected_effect_fingerprint=effect_fingerprint,
            )
        )

    @staticmethod
    async def _await_with_task_lease(
        operation: Awaitable[Any],
        heartbeat: asyncio.Task[None],
        *,
        task: Task | None = None,
        worker_id: str | None = None,
        dispatcher: TaskStoreDispatcher | None = None,
        lease_authority: _DispatchLeaseAuthority | None = None,
    ) -> Any:
        """Supervise one dispatch operation through task-authority settlement."""

        async def run_operation() -> Any:
            return await operation

        operation_task = asyncio.create_task(run_operation())
        owner_cancellation: asyncio.CancelledError | None = None
        heartbeat_failure: BaseException | None = None
        try:
            completed, _pending = await asyncio.wait(
                {operation_task, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError as cancellation:
            if dispatcher is None or task is None or worker_id is None or lease_authority is None:
                operation_task.cancel()
                operation_outcome = await asyncio.gather(
                    operation_task,
                    return_exceptions=True,
                )
                cleanup_failure = operation_outcome[0]
                if isinstance(cleanup_failure, BaseException) and not isinstance(
                    cleanup_failure,
                    asyncio.CancelledError,
                ):
                    if _dispatch_failure_contains_process_control(cleanup_failure):
                        _attach_dispatch_secondary_failure(
                            cleanup_failure,
                            cancellation,
                            label="Dispatch process-control and caller-cancellation evidence",
                        )
                        raise cleanup_failure from exception_cause(cleanup_failure)
                    _attach_dispatch_secondary_failure(
                        cancellation,
                        cleanup_failure,
                        label="Dispatch caller cancellation and cleanup failures",
                    )
                raise cancellation
            owner_cancellation = cancellation
        else:
            if heartbeat not in completed:
                return operation_task.result()
            heartbeat_failure = None if heartbeat.cancelled() else heartbeat.exception()
            if heartbeat_failure is None:
                heartbeat_failure = RuntimeError(
                    "Dispatch task heartbeat stopped without proving task authority."
                )

        if dispatcher is None or task is None or worker_id is None or lease_authority is None:
            # This entrance is retained only for direct internal supervision without a
            # durable queue task.  The production dispatcher always supplies authority.
            operation_outcome = await await_shielded_task_outcome(
                operation_task,
                cancellation=owner_cancellation,
            )
            drain_outcome = _DispatchOperationDrainOutcome(
                operation_failure=operation_outcome.error,
            )
            cancellation_requests_consumed = operation_outcome.cancellation_requests_consumed
            owner_cancellation = operation_outcome.cancellation
        else:
            drain_task = asyncio.create_task(
                dispatcher._drain_dispatch_operation_under_cancellation_fence(
                    operation_task,
                    task=task,
                    worker_id=worker_id,
                    lease_authority=lease_authority,
                    error={
                        "code": (
                            "dispatch_worker_cancelled"
                            if owner_cancellation is not None
                            else "dispatch_task_lease_authority_lost"
                        )
                    },
                )
            )
            shielded_drain = await await_shielded_task_outcome(
                drain_task,
                cancellation=owner_cancellation,
            )
            cancellation_requests_consumed = shielded_drain.cancellation_requests_consumed
            owner_cancellation = shielded_drain.cancellation
            if shielded_drain.error is not None:
                drain_outcome = _DispatchOperationDrainOutcome(
                    operation_failure=shielded_drain.error,
                )
            else:
                if shielded_drain.result is None:  # pragma: no cover - typed task invariant
                    raise RuntimeError("Dispatch cancellation drain returned no outcome.")
                drain_outcome = shielded_drain.result

        if heartbeat_failure is None and heartbeat.done():
            heartbeat_failure = None if heartbeat.cancelled() else heartbeat.exception()
            if heartbeat_failure is None:
                heartbeat_failure = RuntimeError(
                    "Dispatch task heartbeat stopped without proving task authority."
                )

        drain_failures = [
            failure
            for failure in (
                drain_outcome.fence_failure,
                drain_outcome.operation_failure,
                drain_outcome.settlement_failure,
            )
            if failure is not None
        ]
        process_control_failure = next(
            (
                failure
                for failure in drain_failures
                if _dispatch_failure_contains_process_control(failure)
            ),
            None,
        )
        if process_control_failure is not None:
            if heartbeat_failure is not None:
                _attach_dispatch_secondary_failure(
                    process_control_failure,
                    heartbeat_failure,
                    label="Dispatch process-control and lease-loss evidence",
                )
            if owner_cancellation is not None:
                _attach_dispatch_secondary_failure(
                    process_control_failure,
                    owner_cancellation,
                    label="Dispatch process-control and caller-cancellation evidence",
                )
            for failure in drain_failures:
                if failure is not process_control_failure:
                    _attach_dispatch_secondary_failure(
                        process_control_failure,
                        failure,
                        label="Dispatch process-control and drain failures",
                    )
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=owner_cancellation,
            )
            raise process_control_failure from exception_cause(process_control_failure)
        if owner_cancellation is not None:
            if heartbeat_failure is not None:
                _attach_dispatch_secondary_failure(
                    owner_cancellation,
                    heartbeat_failure,
                    label="Dispatch caller cancellation and lease-loss evidence",
                )
            for failure in drain_failures:
                _attach_dispatch_secondary_failure(
                    owner_cancellation,
                    failure,
                    label="Dispatch caller cancellation and drain failures",
                )
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=owner_cancellation,
            )
            raise owner_cancellation from exception_cause(owner_cancellation)

        assert heartbeat_failure is not None
        if drain_failures:
            raise BaseExceptionGroup(
                "Dispatch lease loss and cancellation-fence drain failures",
                [heartbeat_failure, *drain_failures],
            ) from None
        raise heartbeat_failure

    async def _reclaimed_dispatch_handle(
        self,
        task: Task,
        envelope: _QueuedDispatchEnvelope,
    ) -> DispatchHandle:
        status = await self._reclaimed_dispatch_status(
            task_id=task.id,
            envelope=envelope,
        )
        return self._handle(
            envelope.request,
            status,
            queue_task_id=task.id,
            envelope=envelope,
            reclaimed=True,
        )

    @staticmethod
    async def _stop_heartbeat(
        stop: asyncio.Event,
        heartbeat: asyncio.Task[None],
    ) -> None:
        stop.set()
        heartbeat.cancel()
        if heartbeat.done():
            with contextlib.suppress(asyncio.CancelledError, TaskClaimLost):
                heartbeat.result()
            return
        with contextlib.suppress(asyncio.CancelledError, TaskClaimLost):
            await heartbeat

    async def run_worker(
        self,
        runtime: DispatchRuntime,
        *,
        worker_id: str,
        stop: asyncio.Event,
        poll_interval_s: float = 1.0,
        metrics: DurableWorkerMetrics | None = None,
        minimum_idle_delay_s: float | None = None,
        maximum_idle_delay_s: float | None = None,
        idle_backoff_multiplier: float = 2.0,
        idle_jitter_ratio: float = 0.1,
        reconcile_terminal_receipts: bool = True,
        reconciliation_every_s: float = 60.0,
        reclaim_expired_leases: bool = True,
        reclaim_every_s: float = 60.0,
    ) -> None:
        """Claim and run with bounded adaptive demand and independent maintenance."""

        validate_worker_interval(poll_interval_s, "poll_interval_s")
        if metrics is not None and not isinstance(metrics, DurableWorkerMetrics):
            raise TypeError("metrics must be a DurableWorkerMetrics instance.")
        configured_demand_policy = DurableWorkerDemandPolicy(
            dispatch_latency_s=poll_interval_s,
            minimum_idle_delay_s=minimum_idle_delay_s,
            maximum_idle_delay_s=maximum_idle_delay_s,
            backoff_multiplier=idle_backoff_multiplier,
            jitter_ratio=idle_jitter_ratio,
        )
        if type(reconcile_terminal_receipts) is not bool:
            raise TypeError("reconcile_terminal_receipts must be a bool.")
        validate_worker_interval(reconciliation_every_s, "reconciliation_every_s")
        if type(reclaim_expired_leases) is not bool:
            raise TypeError("reclaim_expired_leases must be a bool.")
        validate_worker_interval(reclaim_every_s, "reclaim_every_s")
        durable_runtime = _require_profiled_dispatch_runtime(runtime)
        loop = asyncio.get_running_loop()
        process_next = self.process_next
        uses_base_claim_boundary = (
            getattr(process_next, "__self__", None) is self
            and getattr(process_next, "__func__", None) is TaskStoreDispatcher.process_next
        )
        claim_task_types = self._claim_task_types()
        claim_queries = tuple(TaskQuery(type=task_type) for task_type in claim_task_types)
        if not claim_queries:
            raise ValueError("Queued dispatch workers require at least one task namespace.")
        demand_policy = configured_demand_policy
        if uses_base_claim_boundary and len(claim_queries) > 1:
            per_namespace_maximum_s = poll_interval_s / len(claim_queries)
            if minimum_idle_delay_s is not None and minimum_idle_delay_s > per_namespace_maximum_s:
                raise ValueError(
                    "minimum_idle_delay_s must not exceed poll_interval_s divided by "
                    "the dispatcher task-namespace count."
                )
            demand_policy = DurableWorkerDemandPolicy(
                dispatch_latency_s=poll_interval_s,
                minimum_idle_delay_s=min(
                    configured_demand_policy.minimum_idle_delay_s,
                    per_namespace_maximum_s,
                ),
                maximum_idle_delay_s=min(
                    configured_demand_policy.maximum_idle_delay_s,
                    per_namespace_maximum_s,
                ),
                backoff_multiplier=configured_demand_policy.backoff_multiplier,
                jitter_ratio=configured_demand_policy.jitter_ratio,
            )
        poller = self._tasks._durable_worker_poller(
            claim_queries,
            demand_policy,
            clock=loop.time,
        )
        if metrics is not None:
            poller.set_metrics(metrics)
        reconciliation_cadence = DurableWorkerCadence(reconciliation_every_s)
        reclaim_cadence = DurableWorkerCadence(reclaim_every_s)

        async def reconcile() -> bool:
            settled_before = self._terminal_receipt_reconciliation_settled_count
            reconciliation_generation = self._terminal_receipt_reconciliation_generation
            metrics_token = _RUN_WORKER_METRICS.set(poller.metrics)
            try:
                reconciliation_complete = await self._reconcile_terminal_acknowledgements(
                    durable_runtime
                )
                self._startup_terminal_receipt_reconciliation_pending = (
                    not reconciliation_complete
                    or reconciliation_generation != self._terminal_receipt_reconciliation_generation
                )
            except Exception as exc:
                poller.metrics.store_failure()
                logger.warning(
                    "dispatch terminal acknowledgement discovery failed: error_type=%s error=%s",
                    type(exc).__name__,
                    _safe_runtime_text(durable_runtime, str(exc)),
                )
            finally:
                _RUN_WORKER_METRICS.reset(metrics_token)
            return self._terminal_receipt_reconciliation_settled_count > settled_before

        async def reclaim() -> bool:
            reclaimed_any = False
            for task_type in self._claim_task_types():
                try:
                    reclaimed = await self._tasks.reclaim_expired(query=TaskQuery(type=task_type))
                    reclaimed_any = reclaimed_any or bool(reclaimed)
                except Exception as exc:
                    poller.metrics.store_failure()
                    logger.warning(
                        "dispatch reclaim_expired failed: task_type=%s error_type=%s error=%s",
                        task_type,
                        type(exc).__name__,
                        _safe_runtime_text(durable_runtime, str(exc)),
                    )
            return reclaimed_any

        async def run_step(_now: float, _handled: int) -> DurableWorkerStep:
            meaningful_activity = False
            if reconcile_terminal_receipts:
                reconciliation_ran, reconciled = await reconciliation_cadence.run_if_due(
                    reconcile,
                    now=loop.time(),
                    clock=loop.time,
                )
                if reconciliation_ran:
                    poller.metrics.maintenance(recovery=True)
                meaningful_activity = meaningful_activity or bool(reconciled)
            if reclaim_expired_leases:
                reclaim_ran, reclaimed = await reclaim_cadence.run_if_due(
                    reclaim,
                    now=loop.time(),
                    clock=loop.time,
                )
                if reclaim_ran:
                    poller.metrics.maintenance(reclaim=True)
                meaningful_activity = meaningful_activity or bool(reclaimed)
            if stop.is_set():
                return DurableWorkerStep(stop=True, activity=meaningful_activity)
            reconciliation_generation_before_process = (
                self._terminal_receipt_reconciliation_generation
            )
            suppression_token = _RUN_WORKER_RECONCILIATION_DISPATCHER.set(self)
            poller_token = (
                _RUN_WORKER_DEMAND_POLLER.set((self, poller)) if uses_base_claim_boundary else None
            )
            try:
                if uses_base_claim_boundary:
                    handle = await process_next(runtime, worker_id=worker_id)
                else:
                    override_turn = await poller.claim(
                        lambda: process_next(runtime, worker_id=worker_id),
                        maximum_active_s=self._lease_seconds,
                        store_failure_on_exception=False,
                    )
                    handle = override_turn.value
            except Exception as exc:
                # A transient store error on one task must not kill the durable worker loop.
                logger.error(
                    "dispatch worker failed while processing a task: error_type=%s error=%s",
                    type(exc).__name__,
                    _safe_runtime_text(durable_runtime, str(exc)),
                )
                handle = None
            finally:
                if uses_base_claim_boundary and poller.last_claimed:
                    poller.metrics.handler_finished()
                if poller_token is not None:
                    _RUN_WORKER_DEMAND_POLLER.reset(poller_token)
                _RUN_WORKER_RECONCILIATION_DISPATCHER.reset(suppression_token)
            meaningful_activity = meaningful_activity or poller.last_claimed
            if (
                reconcile_terminal_receipts
                and self._startup_terminal_receipt_reconciliation_pending
                and self._terminal_receipt_reconciliation_generation
                != reconciliation_generation_before_process
            ):
                reconciliation_cadence.expedite()
            # Back off when idle, after a busy-session requeue, or after a lost-lease reclaim —
            # otherwise the just-released/reclaimed task (FIFO-oldest) is re-claimed immediately
            # in a tight loop, re-running the agent with no delay.
            should_wait = (
                handle is None
                or handle.metadata.get("requeued")
                or handle.metadata.get("reclaimed")
            )
            if not should_wait:
                return DurableWorkerStep(
                    handled=1,
                    continue_immediately=True,
                    activity=True,
                )
            wake_deadlines: list[float] = []
            if reconcile_terminal_receipts and reconciliation_cadence.next_run_at is not None:
                wake_deadlines.append(reconciliation_cadence.next_run_at)
            if reclaim_expired_leases and reclaim_cadence.next_run_at is not None:
                wake_deadlines.append(reclaim_cadence.next_run_at)
            return DurableWorkerStep(
                handled=0 if handle is None else 1,
                idle=True,
                next_wake_at=min(wake_deadlines) if wake_deadlines else None,
                activity=meaningful_activity,
            )

        admission_wakeup = None
        try:
            admission_wakeup = await self._tasks._task_admission_wakeup(claim_queries)
            await run_durable_worker_loop(
                run_step,
                poll_interval_s=poll_interval_s,
                stop=stop,
                wait=(
                    wait_or_stop if admission_wakeup is None else admission_wakeup.wait_for_worker
                ),
                demand_policy=demand_policy,
                poller=poller,
                metrics=metrics if metrics is not None else poller.metrics,
            )
        finally:
            if admission_wakeup is not None:
                admission_wakeup.close()
            poller.close()

    async def _acknowledge_terminal_task(
        self,
        runtime: _PreparedSubagentSubmissionRuntime,
        task: Task,
        envelope: _QueuedDispatchEnvelope,
        *,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> DispatchStatus:
        """Release a session receipt only for an exact durable task outcome."""

        try:
            task_type = self._task_type_for_envelope(envelope)
            settlement = await runtime._queued_dispatch_settlement_state(envelope)
            if type(settlement) is not _QueuedDispatchSettlement:
                raise TypeError("Queued dispatch settlement returned an invalid record.")
            session_binding = await _load_dispatch_session_invocation(
                runtime,
                _queued_dispatch_authority_session_id(envelope),
            )
            _require_dispatch_task_authority(
                task,
                envelope=envelope,
                session_binding=session_binding,
                task_type=task_type,
            )
            if task.status is TaskStatus.CANCELLED:
                if not _task_matches_queued_dispatch(
                    task,
                    task_type=task_type,
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
                    task_type=task_type,
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

        requested_generation = self._terminal_receipt_reconciliation_generation
        while self._terminal_receipt_reconciliation_task is not None:
            active_reconciliation = self._terminal_receipt_reconciliation_task
            active_generation = self._terminal_receipt_reconciliation_task_generation
            if active_generation is None:
                raise RuntimeError("Active terminal receipt reconciliation has no generation.")
            if active_reconciliation.get_loop() is not asyncio.get_running_loop():
                if active_reconciliation.done():
                    self._terminal_receipt_reconciliation_task = None
                    self._terminal_receipt_reconciliation_task_generation = None
                    continue
                raise RuntimeError(
                    "TaskStoreDispatcher cannot reconcile terminal receipts from "
                    "multiple event loops concurrently."
                )
            reconciliation_complete = await asyncio.shield(active_reconciliation)
            if not reconciliation_complete or active_generation >= requested_generation:
                return reconciliation_complete
            if self._terminal_receipt_reconciliation_task is active_reconciliation:
                self._terminal_receipt_reconciliation_task = None
                self._terminal_receipt_reconciliation_task_generation = None

        reconciliation_generation = self._terminal_receipt_reconciliation_generation
        reconciliation = asyncio.create_task(
            self._reconcile_terminal_acknowledgements_owned(runtime)
        )
        self._terminal_receipt_reconciliation_task = reconciliation
        self._terminal_receipt_reconciliation_task_generation = reconciliation_generation

        def reconciliation_done(completed: asyncio.Task[bool]) -> None:
            with contextlib.suppress(asyncio.CancelledError):
                completed.exception()
            if self._terminal_receipt_reconciliation_task is completed:
                self._terminal_receipt_reconciliation_task = None
                self._terminal_receipt_reconciliation_task_generation = None

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
                    task_type=task.type,
                )
                try:
                    expected_task_type = (
                        None if envelope is None else self._task_type_for_envelope(envelope)
                    )
                except (TypeError, ValueError):
                    expected_task_type = None
                if (
                    envelope is None
                    or expected_task_type != task.type
                    or (
                        receipt.operation_id != envelope.dispatch_operation_id
                        or receipt.terminal_event_id != envelope.terminal_event_id
                    )
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
                    self._terminal_receipt_reconciliation_settled_count += 1
                except Exception as exc:
                    all_receipts_settled = False
                    worker_metrics = _RUN_WORKER_METRICS.get()
                    if worker_metrics is not None:
                        worker_metrics.store_failure()
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
        tool_capability_ceiling=(
            None
            if request.tool_capability_ceiling is None
            else copy_tool_capability_ceiling(request.tool_capability_ceiling)
        ),
        tool_grants=validate_targeted_tool_grants(request.tool_grants),
        profile_adoption=(
            None
            if request.profile_adoption is None
            else copy_execution_profile_adoption_intent(request.profile_adoption)
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
    runtime: _SessionInvocationRuntime,
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
    envelope: _QueuedDispatchEnvelope,
    session_binding: SessionInvocationBinding,
    task_type: str,
) -> None:
    if not isinstance(task, Task):
        raise TypeError("Dispatch task authority requires a Task.")
    if task.work_contract is not None:
        raise ValueError("Contracted tasks require the verifier-aware execution entrance.")
    if task.type != task_type or task.session_id is not None:
        raise ValueError("Dispatch task structural authority conflicts with its queue.")
    request = envelope.request
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


def _queued_dispatch_authority_session_id(envelope: _QueuedDispatchEnvelope) -> str:
    if envelope.operation_kind == "prepared_subagent":
        if envelope.prepared_subagent is None:
            raise ValueError("Prepared-subagent dispatch has no submission authority.")
        return envelope.prepared_subagent.parent_session_id
    return envelope.request.session_id


def _queued_dispatch_schema_version(
    request: DispatchRequest,
    *,
    exact_fork_source_state_sha256: str | None = None,
) -> Literal[2, 3, 4]:
    """Select the oldest envelope schema that preserves the request authority."""

    if type(request) is not DispatchRequest:
        raise TypeError("Queued dispatch requires an exact DispatchRequest.")
    if exact_fork_source_state_sha256 is not None:
        return _QUEUED_DISPATCH_EXACT_FORK_SCHEMA_VERSION
    if (
        request.tool_capability_ceiling is not None
        or request.tool_grants
        or request.profile_adoption is not None
    ):
        return _QUEUED_DISPATCH_INVOCATION_CONTROL_SCHEMA_VERSION
    return _QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION


def _queued_dispatch_request_payload(
    request: DispatchRequest,
    *,
    schema_version: Literal[2, 3, 4],
) -> dict[str, Any]:
    """Project request authority using the selected durable wire contract."""

    payload = copy_dispatch_request(request).model_dump(mode="json")
    if schema_version == _QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION:
        for field_name in _QUEUED_DISPATCH_INVOCATION_CONTROL_FIELDS:
            payload.pop(field_name)
        return payload
    if schema_version in {
        _QUEUED_DISPATCH_INVOCATION_CONTROL_SCHEMA_VERSION,
        _QUEUED_DISPATCH_EXACT_FORK_SCHEMA_VERSION,
    }:
        return payload
    raise ValueError("Queued dispatch uses an unsupported schema version.")


def _queued_dispatch_request_sha256(
    request: DispatchRequest,
    *,
    schema_version: Literal[2, 3, 4],
) -> str:
    payload = _queued_dispatch_request_payload(request, schema_version=schema_version)
    return sha256(canonical_durable_json_bytes(payload, "queued_dispatch.request")).hexdigest()


def _queued_dispatch_task_id(
    request: DispatchRequest,
    *,
    task_type: str = DEFAULT_DISPATCH_TASK_TYPE,
) -> str:
    """Return one caller-visible dispatch identity across wire namespaces.

    Versioned protocols use separate claim namespaces so older workers cannot
    consume envelopes whose session authority they do not understand. Transport
    versioning must not split the caller's ``dispatch_id`` idempotency scope.
    """

    request = copy_dispatch_request(request)
    task_type = require_clean_nonblank(task_type, "task_type")
    for suffix in (
        INVOCATION_CONTROL_DISPATCH_TASK_TYPE_SUFFIX,
        EXACT_FORK_DISPATCH_TASK_TYPE_SUFFIX,
    ):
        if task_type.endswith(suffix):
            task_type = require_clean_nonblank(
                task_type.removesuffix(suffix),
                "dispatch_identity_task_type",
            )
            break
    return durable_dispatch_queue_task_id(
        task_type=task_type,
        dispatch_id=request.dispatch_id,
    )


def _queued_dispatch_operation_id(
    *,
    queue_task_id: str,
    request: DispatchRequest,
    request_sha256: str,
    session_instance_fingerprint: str,
    source_profile: ExecutionProfileIdentity,
    required_profile: ExecutionProfileIdentity,
    schema_version: Literal[2, 3, 4],
    exact_fork_source_state_sha256: str | None = None,
) -> str:
    material = {
        "record_type": _QUEUED_DISPATCH_RECORD_TYPE,
        "schema_version": schema_version,
        "queue_task_id": require_durable_clean_nonblank(queue_task_id, "queue_task_id"),
        "dispatch_id": request.dispatch_id,
        "session_id": request.session_id,
        "linked_task_id": request.task_id,
        "request_sha256": request_sha256,
        "session_instance_fingerprint": session_instance_fingerprint,
        "source_profile_fingerprint": source_profile.fingerprint,
        "required_profile_fingerprint": required_profile.fingerprint,
    }
    if exact_fork_source_state_sha256 is not None:
        material["exact_fork_source_state_sha256"] = exact_fork_source_state_sha256
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
    exact_fork_source_state_sha256: str | None = None,
) -> _QueuedDispatchEnvelope:
    """Build one immutable envelope from runtime-owned session/profile authority."""

    request = copy_dispatch_request(request)
    schema_version = _queued_dispatch_schema_version(
        request,
        exact_fork_source_state_sha256=exact_fork_source_state_sha256,
    )
    request_sha256 = _queued_dispatch_request_sha256(
        request,
        schema_version=schema_version,
    )
    operation_id = _queued_dispatch_operation_id(
        queue_task_id=queue_task_id,
        request=request,
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        source_profile=source_profile,
        required_profile=required_profile,
        schema_version=schema_version,
        exact_fork_source_state_sha256=exact_fork_source_state_sha256,
    )
    return _QueuedDispatchEnvelope(
        schema_version=schema_version,
        queue_task_id=queue_task_id,
        dispatch_operation_id=operation_id,
        terminal_event_id=_queued_dispatch_terminal_event_id(operation_id),
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        request=request,
        source_profile=source_profile,
        required_profile=required_profile,
        exact_fork_source_state_sha256=exact_fork_source_state_sha256,
    )


def _new_prepared_subagent_dispatch_envelope(
    *,
    intent: DurableSubagentSubmissionIntent,
    session_instance_fingerprint: str,
) -> _QueuedDispatchEnvelope:
    """Build a normal queue envelope whose execution entrance starts a PENDING child."""

    intent = copy_durable_subagent_submission_intent(intent)
    request = _prepared_subagent_dispatch_request(intent)
    request_sha256 = _queued_dispatch_request_sha256(
        request,
        schema_version=_QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION,
    )
    operation_id = _queued_dispatch_operation_id(
        queue_task_id=intent.queue_task_id,
        request=request,
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        source_profile=intent.child_execution_profile,
        required_profile=intent.child_execution_profile,
        schema_version=_QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION,
        exact_fork_source_state_sha256=None,
    )
    return _QueuedDispatchEnvelope(
        schema_version=_QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION,
        queue_task_id=intent.queue_task_id,
        dispatch_operation_id=operation_id,
        terminal_event_id=_queued_dispatch_terminal_event_id(operation_id),
        request_sha256=request_sha256,
        session_instance_fingerprint=session_instance_fingerprint,
        request=request,
        source_profile=intent.child_execution_profile,
        required_profile=intent.child_execution_profile,
        operation_kind="prepared_subagent",
        prepared_subagent=intent,
    )


def _prepared_subagent_dispatch_request(
    intent: DurableSubagentSubmissionIntent,
) -> DispatchRequest:
    """Project bounded routing authority while the envelope owns the full intent."""

    intent = copy_durable_subagent_submission_intent(intent)
    return DispatchRequest(
        session_id=intent.child_session_id,
        messages=[Message.text("user", "Prepared durable subagent dispatch.")],
        dispatch_id=intent.dispatch_id,
        task_id=intent.parent_task_id,
        metadata={
            "durable_subagent": {
                "record_type": "cayu.durable-subagent-dispatch-reference",
                "schema_version": 1,
                "submission_sha256": intent.submission_sha256,
                "child_session_id": intent.child_session_id,
                "queue_task_id": intent.queue_task_id,
            }
        },
        max_steps=1,
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
        and _queued_dispatch_task_input_matches(task.input, envelope)
    )


def _queued_dispatch_task_input_matches(
    task_input: dict[str, Any],
    envelope: _QueuedDispatchEnvelope,
) -> bool:
    """Accept exact in-memory authority or its canonical durable wire representation."""

    if type(task_input) is not dict or set(task_input) != {"dispatch"}:
        return False
    raw_envelope = task_input["dispatch"]
    if type(raw_envelope) is not dict:
        return False
    current = envelope.model_dump(mode="json")
    return raw_envelope == current or raw_envelope == _queued_dispatch_persisted_envelope(envelope)


def _queued_dispatch_persisted_envelope(
    envelope: _QueuedDispatchEnvelope,
) -> dict[str, Any]:
    """Persist only fields defined by the envelope's versioned protocol."""

    persisted = envelope.model_dump(mode="json")
    if envelope.schema_version != _QUEUED_DISPATCH_EXACT_FORK_SCHEMA_VERSION:
        persisted.pop("exact_fork_source_state_sha256")
    if envelope.schema_version == _QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION:
        persisted["request"] = _queued_dispatch_request_payload(
            envelope.request,
            schema_version=envelope.schema_version,
        )
    if (
        envelope.schema_version != _QUEUED_DISPATCH_COMPAT_SCHEMA_VERSION
        or envelope.operation_kind != "resume"
        or envelope.prepared_subagent is not None
    ):
        return persisted
    return {
        key: value
        for key, value in persisted.items()
        if key not in {"operation_kind", "prepared_subagent"}
    }


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

    prepared_invocation = session_request_boundary.prepare_resume_request(
        ResumeRequest(
            session_id=request.session_id,
            messages=request.messages,
            target=request.target,
            tool_capability_ceiling=request.tool_capability_ceiling,
            tool_grants=request.tool_grants,
            profile_adoption=request.profile_adoption,
            metadata=request.metadata,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
            thinking=request.thinking,
        ),
        redactor=redactor,
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
        tool_capability_ceiling=prepared_invocation.tool_capability_ceiling,
        tool_grants=prepared_invocation.tool_grants,
        profile_adoption=prepared_invocation.profile_adoption,
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


def _require_prepared_subagent_submission_runtime(
    runtime: object,
) -> _PreparedSubagentSubmissionRuntime:
    """Reject prepared publication without its three narrow session operations."""

    for method_name in (
        "session_invocation_for_dispatch",
        "_queued_dispatch_settlement_state",
        "_acknowledge_queued_dispatch",
    ):
        try:
            method = getattr(runtime, method_name)
        except (AttributeError, TypeError):
            raise TypeError(f"Prepared subagent runtime {method_name} must be callable.") from None
        if not callable(method):
            raise TypeError(f"Prepared subagent runtime {method_name} must be callable.")
    return cast("_PreparedSubagentSubmissionRuntime", runtime)


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
