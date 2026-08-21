from __future__ import annotations

import asyncio
import base64
import json
import math
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right, insort
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime, utc_clock
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    revalidate_model_input,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.runtime.aggregates import EXACT_AGGREGATE, AggregateAccuracy, AggregateCount
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    SessionInvocationBinding,
    TaskExecutionSource,
    TaskInvocation,
    copy_invocation_origin,
    copy_invocation_origin_claim,
    copy_session_invocation_binding,
    copy_task_invocation,
    inherited_task_invocation,
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.runtime.work_contracts import (
    WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES,
    WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS,
    WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
    WORK_CONTRACT_TASK_CREATION_MAX_ITEMS,
    WORK_CONTRACT_TASK_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_ITEMS,
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractRef,
    completion_decision_application_request_sha256,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_request_sha256,
    copy_completion_decision_application_request,
    copy_completion_decision_create,
    copy_completion_proposal_create,
    copy_completion_verification_claim_request,
    copy_work_attempt_create,
    copy_work_contract,
    copy_work_contract_ref,
    preflight_work_completion_document,
    require_bounded_work_completion_document,
    validate_completion_decision_contract,
    validate_work_completion_idempotency_key,
    validate_work_completion_linked_id,
    work_attempt_request_sha256,
)

_TASK_RETRY_COST_MAX_DIGITS = 64
_TASK_RETRY_TOTAL_COST_MAX_DIGITS = 128
_TASK_RETRY_COST_MAX_DECIMAL_PLACES = 64
_TASK_RETRY_MAX_ATTEMPT_TOKEN_REPORT = MAX_DURABLE_JSON_INTEGER // 100
_TASK_RETRY_CANCELLATION_REQUESTED_REASON = "retry_cancellation_requested"


def _bounded_task_retry_decimal(
    value: Decimal,
    field_name: str,
    *,
    max_digits: int,
) -> Decimal:
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal.")
    digits = value.as_tuple().digits
    exponent = value.as_tuple().exponent
    if len(digits) > max_digits:
        raise ValueError(f"{field_name} exceeds its decimal digit limit.")
    if (
        not isinstance(exponent, int)
        or exponent < -_TASK_RETRY_COST_MAX_DECIMAL_PLACES
        or exponent > _TASK_RETRY_COST_MAX_DIGITS
    ):
        raise ValueError(f"{field_name} exceeds its decimal scale limit.")
    return value


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskClaimLost(ValueError):
    """A worker no longer owns the active lease required for a task mutation."""


class TaskTerminalizationConflict(ValueError):
    """An idempotency key is already bound to another terminalization intent."""


class TaskTerminalKind(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRetryAttemptDisposition(StrEnum):
    """Application-classified outcome for one retry-series attempt."""

    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    NON_RETRYABLE_FAILURE = "non_retryable_failure"
    CANCELLED = "cancelled"


class TaskRetrySeriesDisposition(StrEnum):
    """Durable retry-series state or terminal reason."""

    ACTIVE = "active"
    RETRY_SCHEDULED = "retry_scheduled"
    SUCCEEDED = "succeeded"
    NON_RETRYABLE_FAILURE = "non_retryable_failure"
    CANCELLED = "cancelled"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    ELAPSED_EXHAUSTED = "elapsed_exhausted"
    TOKENS_EXHAUSTED = "tokens_exhausted"
    COST_EXHAUSTED = "cost_exhausted"


class TaskRetryEventType(StrEnum):
    """Stable task-level retry event types committed with a settlement."""

    ATTEMPT_SETTLED = "task.retry.attempt_settled"
    RETRY_SCHEDULED = "task.retry.scheduled"
    SERIES_TERMINAL = "task.retry.series_terminal"


def _validate_task_retry_cost_currency(value: str) -> str:
    value = require_clean_nonblank(value, "cost_currency").upper()
    if len(value.encode("utf-8")) > 16:
        raise ValueError("cost_currency must be at most 16 UTF-8 bytes.")
    return value


class TaskRetryPolicy(BaseModel):
    """Serializable cumulative limits and backoff for one task retry series."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    max_attempts: StrictInt = Field(ge=1, le=100)
    max_elapsed_seconds: StrictFloat | None = Field(default=None, gt=0, le=31_536_000)
    max_total_tokens: StrictInt | None = Field(
        default=None,
        gt=0,
        le=MAX_DURABLE_JSON_INTEGER,
        description=(
            "Maximum cumulative reported tokens used for retry-successor admission; "
            "not an external-dispatch reservation."
        ),
    )
    max_estimated_cost: Decimal | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum cumulative reported estimated cost used for retry-successor "
            "admission; not an external-dispatch reservation."
        ),
    )
    cost_currency: str = "USD"
    initial_backoff_seconds: StrictFloat = Field(default=1.0, ge=0, le=86_400)
    backoff_multiplier: StrictFloat = Field(default=2.0, ge=1, le=100)
    max_backoff_seconds: StrictFloat = Field(default=300.0, ge=0, le=86_400)

    @model_validator(mode="after")
    def validate_bounds(self) -> TaskRetryPolicy:
        for field_name in (
            "max_elapsed_seconds",
            "initial_backoff_seconds",
            "backoff_multiplier",
            "max_backoff_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        return self

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_max_estimated_cost(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        return _bounded_task_retry_decimal(
            value,
            "max_estimated_cost",
            max_digits=_TASK_RETRY_COST_MAX_DIGITS,
        )

    @field_validator("cost_currency")
    @classmethod
    def validate_cost_currency(cls, value: str) -> str:
        return _validate_task_retry_cost_currency(value)


class TaskRetrySeriesSnapshot(BaseModel):
    """Bounded cumulative retry authority carried by each durable attempt."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    series_id: str
    causal_budget_id: str
    authority_sha256: str
    attempt: StrictInt = Field(ge=1, le=100)
    policy: TaskRetryPolicy
    started_at: datetime
    cumulative_tokens: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cumulative_estimated_cost: Decimal = Field(default=Decimal(0), ge=0)
    attempts_remaining: StrictInt = Field(ge=0, le=99)
    tokens_remaining: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    estimated_cost_remaining: Decimal | None = Field(default=None, ge=0)
    elapsed_deadline: datetime | None = None
    disposition: TaskRetrySeriesDisposition = TaskRetrySeriesDisposition.ACTIVE
    predecessor_task_id: str | None = None
    successor_task_id: str | None = None
    next_eligible_at: datetime | None = None

    @field_validator("series_id", "causal_budget_id")
    @classmethod
    def validate_series_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("authority_sha256")
    @classmethod
    def validate_authority_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("authority_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("predecessor_task_id", "successor_task_id")
    @classmethod
    def validate_optional_task_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "started_at")

    @field_validator("next_eligible_at")
    @classmethod
    def normalize_next_eligible_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "next_eligible_at")

    @field_validator("elapsed_deadline")
    @classmethod
    def normalize_elapsed_deadline(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "elapsed_deadline")

    @field_validator("cumulative_estimated_cost", "estimated_cost_remaining")
    @classmethod
    def validate_estimated_costs(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        return _bounded_task_retry_decimal(
            value,
            info.field_name,
            max_digits=(
                _TASK_RETRY_TOTAL_COST_MAX_DIGITS
                if info.field_name == "cumulative_estimated_cost"
                else _TASK_RETRY_COST_MAX_DIGITS
            ),
        )

    @model_validator(mode="after")
    def validate_snapshot(self) -> TaskRetrySeriesSnapshot:
        if self.attempts_remaining != max(0, self.policy.max_attempts - self.attempt):
            raise ValueError("attempts_remaining conflicts with the retry policy.")
        expected_tokens = (
            None
            if self.policy.max_total_tokens is None
            else max(0, self.policy.max_total_tokens - self.cumulative_tokens)
        )
        if self.tokens_remaining != expected_tokens:
            raise ValueError("tokens_remaining conflicts with cumulative token usage.")
        expected_cost = (
            None
            if self.policy.max_estimated_cost is None
            else max(
                Decimal(0),
                self.policy.max_estimated_cost - self.cumulative_estimated_cost,
            )
        )
        if self.estimated_cost_remaining != expected_cost:
            raise ValueError("estimated_cost_remaining conflicts with cumulative cost.")
        expected_deadline = (
            None
            if self.policy.max_elapsed_seconds is None
            else self.started_at + timedelta(seconds=self.policy.max_elapsed_seconds)
        )
        if self.elapsed_deadline != expected_deadline:
            raise ValueError("elapsed_deadline conflicts with the retry policy.")
        scheduled = self.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        if scheduled != (self.successor_task_id is not None):
            raise ValueError("Only retry_scheduled snapshots carry a successor_task_id.")
        if scheduled != (self.next_eligible_at is not None):
            raise ValueError("Only retry_scheduled snapshots carry next_eligible_at.")
        return self


class TaskRetryEvent(BaseModel):
    """Bounded, failure-payload-free retry evidence committed by a task store."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    type: TaskRetryEventType
    task_id: str
    series_id: str
    causal_budget_id: str
    attempt: StrictInt = Field(ge=1, le=100)
    disposition: TaskRetrySeriesDisposition
    occurred_at: datetime
    attempts_remaining: StrictInt = Field(ge=0, le=99)
    tokens_remaining: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    estimated_cost_remaining: Decimal | None = Field(default=None, ge=0)
    cost_currency: str
    elapsed_deadline: datetime | None = None
    next_eligible_at: datetime | None = None

    @field_validator("id", "task_id", "series_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("cost_currency")
    @classmethod
    def validate_cost_currency(cls, value: str) -> str:
        return _validate_task_retry_cost_currency(value)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @field_validator("elapsed_deadline", "next_eligible_at")
    @classmethod
    def normalize_optional_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("estimated_cost_remaining")
    @classmethod
    def validate_estimated_cost_remaining(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        return _bounded_task_retry_decimal(
            value,
            "estimated_cost_remaining",
            max_digits=_TASK_RETRY_COST_MAX_DIGITS,
        )


TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES = 256
_CONTRACT_TASK_JSON_FIELDS = ("input", "metadata", "status_payload", "result", "error")


def _preflight_bounded_task_payloads(
    value: object,
    field_names: tuple[str, ...] = _CONTRACT_TASK_JSON_FIELDS,
    *,
    field_label: str = "Contract-bound task",
) -> None:
    document = cast("dict[str, object]", value) if type(value) is dict else None
    for field_name in field_names:
        field_value = (
            document.get(field_name) if document is not None else getattr(value, field_name)
        )
        if field_value is None:
            continue
        preflight_work_completion_document(
            field_value,
            f"{field_label} {field_name}",
            max_bytes=WORK_CONTRACT_TASK_MAX_BYTES,
            max_items=WORK_CONTRACT_TASK_MAX_ITEMS,
        )


class TaskOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"


class Task(BaseModel):
    """Durable unit of work.

    Tasks are intentionally generic. They can represent background jobs,
    workflow steps, external work items, orchestrator assignments, or a
    single-agent durable job.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    title: str | None = None
    description: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    available_at: datetime | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    status_reason: str | None = None
    status_payload: dict[str, Any] | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    invocation: TaskInvocation = Field(frozen=True)
    retry_series: TaskRetrySeriesSnapshot | None = None
    work_contract: WorkContractRef | None = Field(default=None, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def preflight_work_contract_payloads(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        document = cast("dict[str, object]", value)
        if document.get("work_contract") is None:
            return value
        _preflight_bounded_task_payloads(document)
        return value

    @field_validator("input", "metadata", mode="before")
    @classmethod
    def copy_json_object(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_durable_json_object(value, info.field_name)

    @field_validator("status_payload", "result", "error", mode="before")
    @classmethod
    def copy_optional_json_object(
        cls,
        value: dict[str, Any] | None,
        info,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @field_validator("id", "type")
    @classmethod
    def validate_nonblank_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "title",
        "description",
        "session_id",
        "parent_task_id",
        "assigned_agent_name",
        "worker_id",
        "status_reason",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name in {"title", "description", "status_reason"}:
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("available_at")
    @classmethod
    def normalize_available_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "available_at")

    @field_validator("work_contract", mode="before")
    @classmethod
    def copy_work_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @model_validator(mode="after")
    def validate_retry_and_work_contract_authority(self) -> Task:
        if self.retry_series is not None:
            expected = _task_retry_attempt_authority_sha256(
                task_id=self.id,
                task_type=self.type,
                title=self.title,
                description=self.description,
                parent_task_id=self.parent_task_id,
                assigned_agent_name=self.assigned_agent_name,
                available_at=self.available_at,
                created_at=self.created_at,
                task_input=self.input,
                metadata=self.metadata,
                invocation=self.invocation,
                series_id=self.retry_series.series_id,
                causal_budget_id=self.retry_series.causal_budget_id,
                attempt=self.retry_series.attempt,
                policy=self.retry_series.policy,
                started_at=self.retry_series.started_at,
                cumulative_tokens=self.retry_series.cumulative_tokens,
                cumulative_estimated_cost=self.retry_series.cumulative_estimated_cost,
                predecessor_task_id=self.retry_series.predecessor_task_id,
            )
            if self.retry_series.authority_sha256 != expected:
                raise ValueError("Task retry-series authority conflicts with its task evidence.")
        if self.retry_series is not None and self.work_contract is not None:
            raise ValueError("Retry-series tasks cannot use verified work contracts.")
        if self.work_contract is None:
            return self
        validate_work_completion_linked_id(self.id, "id")
        if self.session_id is not None:
            validate_work_completion_linked_id(self.session_id, "session_id")
        if self.worker_id is not None:
            validate_work_completion_linked_id(self.worker_id, "worker_id")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Contract-bound task",
            max_bytes=WORK_CONTRACT_TASK_MAX_BYTES,
            max_items=WORK_CONTRACT_TASK_MAX_ITEMS,
        )
        return self


class TaskInvocationSnapshot(BaseModel):
    """Bounded task identity and immutable provenance for delegation boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    session_id: str | None
    invocation: TaskInvocation

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "id")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "session_id")

    @field_validator("invocation")
    @classmethod
    def copy_invocation(cls, value: TaskInvocation) -> TaskInvocation:
        return copy_task_invocation(value)


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str | None = None
    type: str
    title: str | None = None
    description: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    available_at: datetime | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retry_policy: TaskRetryPolicy | None = None
    work_contract: WorkContractRef | None = None
    invocation_origin: InvocationOriginClaim | None = None
    _verified_invocation_origin: InvocationOrigin | None = PrivateAttr(default=None)
    _runtime_invocation_source: TaskExecutionSource | None = PrivateAttr(default=None)
    _runtime_session_binding: SessionInvocationBinding | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def preflight_work_contract_payloads(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        document = cast("dict[str, object]", value)
        if document.get("work_contract") is None:
            return value
        _preflight_bounded_task_payloads(document, ("input", "metadata"))
        return value

    @field_validator("input", "metadata", mode="before")
    @classmethod
    def copy_json_object(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_durable_json_object(value, info.field_name)

    @field_validator("type")
    @classmethod
    def validate_nonblank_type(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "task_id",
        "title",
        "description",
        "session_id",
        "parent_task_id",
        "assigned_agent_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name in {"title", "description"}:
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("available_at")
    @classmethod
    def normalize_available_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "available_at")

    @field_validator("work_contract", mode="before")
    @classmethod
    def copy_work_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @model_validator(mode="after")
    def validate_retry_and_work_contract_shape(self) -> TaskCreate:
        if self.retry_policy is not None:
            if self.session_id is not None:
                raise ValueError("Retry-series tasks must start as unattached queue work.")
            if self.work_contract is not None:
                raise ValueError("Retry-series tasks cannot use verified work contracts.")
        if self.work_contract is None:
            return self
        if self.task_id is not None:
            validate_work_completion_linked_id(self.task_id, "task_id")
        if self.session_id is not None:
            validate_work_completion_linked_id(self.session_id, "session_id")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Contract-bound task creation request",
            max_bytes=WORK_CONTRACT_TASK_MAX_BYTES,
            max_items=WORK_CONTRACT_TASK_MAX_ITEMS,
        )
        return self


class TaskTerminalizationRequest(BaseModel):
    """One claim-fenced, replay-safe completion or failure intent."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str
    worker_id: str
    kind: TaskTerminalKind
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    idempotency_key: str

    @field_validator("task_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("result", "error", mode="before")
    @classmethod
    def copy_payload(
        cls,
        value: dict[str, Any] | None,
        info,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> TaskTerminalizationRequest:
        if self.kind is TaskTerminalKind.COMPLETED:
            if self.result is None or self.error is not None:
                raise ValueError("Completed terminalization requires result and forbids error.")
        elif self.error is None or self.result is not None:
            raise ValueError("Failed terminalization requires error and forbids result.")
        return self


class TaskTerminalizationReceipt(BaseModel):
    """Immutable commit evidence for one task terminalization intent."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str
    idempotency_key: str
    worker_id: str
    kind: TaskTerminalKind
    request_sha256: str
    task: Task
    committed_at: datetime

    @field_validator("task_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def copy_terminal_task(cls, value: Task) -> Task:
        if type(value) is not Task:
            raise TypeError("task must be a Task instance.")
        return Task.model_validate(value.model_dump(mode="python"))

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_receipt_task(self) -> TaskTerminalizationReceipt:
        expected_status = (
            TaskStatus.COMPLETED if self.kind is TaskTerminalKind.COMPLETED else TaskStatus.FAILED
        )
        if self.task.id != self.task_id or self.task.status is not expected_status:
            raise ValueError("Terminalization receipt conflicts with its terminal task.")
        if self.task.worker_id is not None or self.task.lease_expires_at is not None:
            raise ValueError("Terminalization receipt task retains live claim ownership.")
        return self


class _TaskRetryAttemptOutcome(BaseModel):
    """Shared validated application-owned outcome material."""

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    idempotency_key: str
    disposition: TaskRetryAttemptDisposition
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    token_count: StrictInt = Field(
        default=0,
        ge=0,
        le=_TASK_RETRY_MAX_ATTEMPT_TOKEN_REPORT,
    )
    estimated_cost: Decimal = Field(default=Decimal(0), ge=0)
    retry_after_seconds: StrictFloat | None = Field(default=None, ge=0, le=86_400)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("result", "error", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any] | None, info) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @field_validator("estimated_cost")
    @classmethod
    def validate_estimated_cost(cls, value: Decimal) -> Decimal:
        return _bounded_task_retry_decimal(
            value,
            "estimated_cost",
            max_digits=_TASK_RETRY_COST_MAX_DIGITS,
        )

    @model_validator(mode="after")
    def validate_disposition_payload(self) -> _TaskRetryAttemptOutcome:
        if self.retry_after_seconds is not None and not math.isfinite(self.retry_after_seconds):
            raise ValueError("retry_after_seconds must be finite.")
        if self.disposition is TaskRetryAttemptDisposition.SUCCEEDED:
            if self.result is None or self.error is not None:
                raise ValueError("Succeeded attempts require result and forbid error.")
        elif self.result is not None or self.error is None:
            raise ValueError("Non-success attempts require error and forbid result.")
        if (
            self.retry_after_seconds is not None
            and self.disposition is not TaskRetryAttemptDisposition.RETRYABLE_FAILURE
        ):
            raise ValueError("Only retryable failures accept retry_after_seconds.")
        return self


class TaskRetrySettlementRequest(_TaskRetryAttemptOutcome):
    """One typed, claim-fenced retry-attempt outcome."""

    task_id: str
    worker_id: str
    causal_budget_id: str

    @field_validator("task_id", "worker_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class TaskRetryAttemptReport(_TaskRetryAttemptOutcome):
    """Application-owned classification and bounded accounting for one attempt."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


class TaskRetrySettlementResult(BaseModel):
    """Immutable settlement receipt for one attempt and optional successor."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    idempotency_key: str
    request_sha256: str
    task: Task
    successor: Task | None = None
    events: tuple[TaskRetryEvent, ...] = Field(min_length=2, max_length=2)
    committed_at: datetime

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "task_id")

    @field_validator("idempotency_key")
    @classmethod
    def validate_result_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_settlement_evidence(self) -> TaskRetrySettlementResult:
        series = self.task.retry_series
        if self.task.id != self.task_id or series is None:
            raise ValueError("Task retry receipt conflicts with its settled attempt.")
        if self.task.worker_id is not None or self.task.lease_expires_at is not None:
            raise ValueError("Task retry receipt retains live attempt ownership.")
        expected_status = {
            TaskRetrySeriesDisposition.SUCCEEDED: TaskStatus.COMPLETED,
            TaskRetrySeriesDisposition.CANCELLED: TaskStatus.CANCELLED,
        }.get(series.disposition, TaskStatus.FAILED)
        if self.task.status is not expected_status:
            raise ValueError("Task retry receipt conflicts with its series disposition.")

        scheduled = series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        if scheduled != (self.successor is not None):
            raise ValueError("Task retry receipt has contradictory successor evidence.")
        if self.successor is not None:
            successor_series = self.successor.retry_series
            if (
                self.successor.id != series.successor_task_id
                or self.successor.status is not TaskStatus.PENDING
                or self.successor.available_at != series.next_eligible_at
                or successor_series is None
                or successor_series.series_id != series.series_id
                or successor_series.attempt != series.attempt + 1
                or successor_series.predecessor_task_id != self.task.id
                or successor_series.disposition is not TaskRetrySeriesDisposition.ACTIVE
                or successor_series.policy != series.policy
                or successor_series.started_at != series.started_at
                or successor_series.causal_budget_id != series.causal_budget_id
                or successor_series.cumulative_tokens != series.cumulative_tokens
                or successor_series.cumulative_estimated_cost != series.cumulative_estimated_cost
                or successor_series.tokens_remaining != series.tokens_remaining
                or successor_series.estimated_cost_remaining != series.estimated_cost_remaining
                or successor_series.elapsed_deadline != series.elapsed_deadline
                or self.successor.type != self.task.type
                or self.successor.title != self.task.title
                or self.successor.description != self.task.description
                or self.successor.parent_task_id != self.task.parent_task_id
                or self.successor.assigned_agent_name != self.task.assigned_agent_name
                or self.successor.input != self.task.input
                or self.successor.metadata != self.task.metadata
                or self.successor.invocation != self.task.invocation
                or self.successor.session_id is not None
                or self.successor.worker_id is not None
                or self.successor.lease_expires_at is not None
                or self.successor.status_reason is not None
                or self.successor.status_payload is not None
                or self.successor.result is not None
                or self.successor.error is not None
                or self.successor.started_at is not None
                or self.successor.completed_at is not None
                or self.successor.created_at != self.committed_at
                or self.successor.updated_at != self.committed_at
            ):
                raise ValueError("Task retry receipt successor conflicts with the settled attempt.")

        expected_event_types = (
            TaskRetryEventType.ATTEMPT_SETTLED,
            (
                TaskRetryEventType.RETRY_SCHEDULED
                if scheduled
                else TaskRetryEventType.SERIES_TERMINAL
            ),
        )
        for event, expected_type in zip(self.events, expected_event_types, strict=True):
            if (
                event.type is not expected_type
                or event.task_id != self.task_id
                or event.series_id != series.series_id
                or event.causal_budget_id != series.causal_budget_id
                or event.attempt != series.attempt
                or event.disposition is not series.disposition
                or event.occurred_at != self.committed_at
                or event.attempts_remaining != series.attempts_remaining
                or event.tokens_remaining != series.tokens_remaining
                or event.estimated_cost_remaining != series.estimated_cost_remaining
                or event.cost_currency != series.policy.cost_currency
                or event.elapsed_deadline != series.elapsed_deadline
                or event.next_eligible_at != series.next_eligible_at
            ):
                raise ValueError("Task retry receipt event conflicts with the settled attempt.")
        return self


class CompletionDecisionApplicationReceipt(BaseModel):
    """Immutable evidence that one verifier decision was applied to its task."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    decision_id: str
    idempotency_key: str
    request_sha256: str
    task: Task
    applied_at: datetime

    @field_validator("task_id", "decision_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        if info.field_name == "idempotency_key":
            return validate_work_completion_idempotency_key(value)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def copy_task(cls, value: object) -> object:
        if type(value) is Task:
            _preflight_bounded_task_payloads(
                value,
                field_label="Decision-application receipt task",
            )
        return revalidate_model_input(value, Task)

    @field_validator("applied_at")
    @classmethod
    def normalize_applied_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "applied_at")

    @model_validator(mode="after")
    def validate_receipt_task(self) -> CompletionDecisionApplicationReceipt:
        if self.task.id != self.task_id:
            raise ValueError("Decision-application receipt conflicts with its task.")
        if self.task.work_contract is None:
            raise ValueError("Decision-application receipt requires a contract-bound task.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Completion decision application receipt",
            max_bytes=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES,
            max_items=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS,
        )
        return self


class TaskTerminalizationRetryPolicy(BaseModel):
    """Finite retry and backoff bounds for acknowledgement-ambiguous writes."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    max_attempts: StrictInt = Field(default=3, ge=1, le=10)
    attempt_timeout_seconds: StrictFloat = Field(default=30.0, gt=0, le=300)
    initial_backoff_seconds: StrictFloat = Field(default=0.05, ge=0, le=60)
    backoff_multiplier: StrictFloat = Field(default=2.0, ge=1, le=10)
    max_backoff_seconds: StrictFloat = Field(default=1.0, ge=0, le=60)


class TaskTerminalizationRetryResult(BaseModel):
    """Detached terminal task plus observable retry/reconciliation evidence."""

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    task: Task
    attempt_count: StrictInt = Field(ge=1, le=10)
    receipt_reconciled: StrictBool
    elapsed_seconds: StrictFloat = Field(default=0.0, ge=0)
    applied_backoff_seconds: StrictFloat = Field(default=0.0, ge=0)

    @field_validator("task", mode="before")
    @classmethod
    def copy_terminal_task(cls, value: Task) -> Task:
        if type(value) is not Task:
            raise TypeError("task must be a Task instance.")
        return Task.model_validate(value.model_dump(mode="python"))


class TaskTerminalizationUncertain(RuntimeError):
    """Bounded evidence that no exact terminalization receipt was observed."""

    def __init__(
        self,
        *,
        task_id: str,
        idempotency_key: str,
        attempt_count: int,
        error_category: str,
        elapsed_seconds: float = 0.0,
        applied_backoff_seconds: float = 0.0,
    ) -> None:
        for field_name, value in (
            ("elapsed_seconds", elapsed_seconds),
            ("applied_backoff_seconds", applied_backoff_seconds),
        ):
            if type(value) is not float:
                raise TypeError(f"{field_name} must be a float.")
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative.")
        self.task_id = _bounded_task_terminalization_evidence(task_id)
        self.idempotency_key = _bounded_task_terminalization_evidence(idempotency_key)
        self.attempt_count = attempt_count
        self.error_category = error_category
        self.elapsed_seconds = elapsed_seconds
        self.applied_backoff_seconds = applied_backoff_seconds
        super().__init__(
            "Task terminalization outcome is uncertain for "
            f"task {self.task_id} after {attempt_count} attempts "
            f"(category={error_category})."
        )


class TaskQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: TaskStatus | None = None
    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    order_by: TaskOrder = TaskOrder.UPDATED_AT_DESC

    @field_validator("q", "type", "session_id", "parent_task_id", "assigned_agent_name")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class TaskAggregateFilter(BaseModel):
    """Current task attributes that may scope a store-native aggregate."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None

    @field_validator("type", "session_id", "parent_task_id", "assigned_agent_name")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class TaskStatusCounts(BaseModel):
    """Complete current-task counts for every lifecycle status."""

    model_config = ConfigDict(extra="forbid")

    pending: AggregateCount = Field(ge=0)
    claimed: AggregateCount = Field(ge=0)
    running: AggregateCount = Field(ge=0)
    paused: AggregateCount = Field(ge=0)
    blocked: AggregateCount = Field(ge=0)
    needs_attention: AggregateCount = Field(ge=0)
    completed: AggregateCount = Field(ge=0)
    failed: AggregateCount = Field(ge=0)
    cancelled: AggregateCount = Field(ge=0)


class TaskOperationalSnapshot(BaseModel):
    """Exact current task counts captured by one store-local read snapshot."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    total_count: AggregateCount = Field(ge=0)
    counts_by_status: TaskStatusCounts
    claimable_pending_count: AggregateCount = Field(ge=0)
    scheduled_pending_count: AggregateCount = Field(ge=0)
    accuracy: AggregateAccuracy

    @field_validator("counts_by_status")
    @classmethod
    def copy_counts_by_status(cls, value: TaskStatusCounts) -> TaskStatusCounts:
        return TaskStatusCounts.model_validate(value.model_dump(mode="python", warnings=False))

    @field_validator("accuracy")
    @classmethod
    def copy_accuracy(cls, value: AggregateAccuracy) -> AggregateAccuracy:
        return AggregateAccuracy.model_validate(value.model_dump(mode="python", warnings=False))

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_total(self) -> TaskOperationalSnapshot:
        if sum(self.counts_by_status.model_dump().values()) != self.total_count:
            raise ValueError("Task status counts must sum to total_count.")
        if (
            self.claimable_pending_count + self.scheduled_pending_count
            > self.counts_by_status.pending
        ):
            raise ValueError("Claimable and scheduled pending counts cannot exceed pending count.")
        return self


TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS = 50
TASK_TOPOLOGY_MAX_EXPANDED_PARENTS = 50
TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT = 25
TASK_TOPOLOGY_MAX_BRANCH_LIMIT = 100
TASK_TOPOLOGY_MAX_NODES = 500
TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES = 1024
TASK_TOPOLOGY_MAX_CURSOR_BYTES = 4096
TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES = 4096
TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH = 128
TASK_TOPOLOGY_MAX_VALIDATION_NODES = 4096
TaskTopologyTruncatedField = Literal[
    "type",
    "title",
    "assigned_agent_name",
    "status_reason",
]


class TaskTopologyCycle(ValueError):
    """Durable parent-task records contain a cycle reachable from the projection."""


class TaskTopologyInconsistent(ValueError):
    """Durable task records cannot form a truthful bounded topology projection."""


class TaskTopologyTraversalLimitExceeded(ValueError):
    """Task ancestry cannot be validated within the bounded topology contract."""


def _bounded_task_topology_text(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
    allow_controls: bool = False,
) -> str:
    validator = require_nonblank if allow_controls else require_clean_nonblank
    value = validator(value, field_name)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain portable Unicode text.") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds the task topology byte limit.")
    return value


def _bounded_task_topology_display(
    value: str | None,
    field_name: TaskTopologyTruncatedField,
    *,
    allow_controls: bool,
) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    try:
        return (
            _bounded_task_topology_text(
                value,
                field_name,
                max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
                allow_controls=allow_controls,
            ),
            False,
        )
    except ValueError:
        # Oversized display text is omitted rather than copied into the bounded
        # projection. The explicit marker keeps absence distinct from truncation.
        if len(value.encode("utf-8", errors="surrogatepass")) > (
            TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES
        ):
            return None, True
        raise


class TaskTopologyNode(BaseModel):
    """Payload-free bounded identity for one task topology node."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str | None
    title: str | None
    status: TaskStatus
    status_reason: str | None
    session_id: str | None
    parent_task_id: str | None
    assigned_agent_name: str | None
    created_at: datetime
    updated_at: datetime
    truncated_fields: tuple[TaskTopologyTruncatedField, ...] = ()

    @classmethod
    def from_task(cls, task: Task) -> TaskTopologyNode:
        if type(task) is not Task:
            raise TypeError("Task topology nodes require Task instances.")
        task_type, type_truncated = _bounded_task_topology_display(
            task.type,
            "type",
            allow_controls=False,
        )
        title, title_truncated = _bounded_task_topology_display(
            task.title,
            "title",
            allow_controls=True,
        )
        assigned_agent_name, agent_truncated = _bounded_task_topology_display(
            task.assigned_agent_name,
            "assigned_agent_name",
            allow_controls=False,
        )
        status_reason, reason_truncated = _bounded_task_topology_display(
            task.status_reason,
            "status_reason",
            allow_controls=True,
        )
        truncated_fields = tuple(
            field_name
            for field_name, truncated in (
                ("type", type_truncated),
                ("title", title_truncated),
                ("assigned_agent_name", agent_truncated),
                ("status_reason", reason_truncated),
            )
            if truncated
        )
        try:
            return cls(
                id=task.id,
                type=task_type,
                title=title,
                status=task.status,
                status_reason=status_reason,
                session_id=task.session_id,
                parent_task_id=task.parent_task_id,
                assigned_agent_name=assigned_agent_name,
                created_at=task.created_at,
                updated_at=task.updated_at,
                truncated_fields=truncated_fields,
            )
        except (TypeError, ValueError) as exc:
            raise TaskTopologyInconsistent(
                "A task record cannot be represented by the bounded topology contract."
            ) from exc

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("session_id", "parent_task_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("type", "assigned_agent_name")
    @classmethod
    def validate_optional_clean_display(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
        )

    @field_validator("title", "status_reason")
    @classmethod
    def validate_optional_display(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
            allow_controls=True,
        )

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("truncated_fields")
    @classmethod
    def validate_truncated_fields(
        cls,
        value: tuple[TaskTopologyTruncatedField, ...],
    ) -> tuple[TaskTopologyTruncatedField, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Task topology truncated_fields must not contain duplicates.")
        canonical = ("type", "title", "assigned_agent_name", "status_reason")
        if tuple(field for field in canonical if field in value) != value:
            raise ValueError("Task topology truncated_fields must use canonical order.")
        return value

    @model_validator(mode="after")
    def validate_display_omissions(self) -> TaskTopologyNode:
        truncated = set(self.truncated_fields)
        for field_name in truncated:
            if getattr(self, field_name) is not None:
                raise ValueError("Truncated task topology display fields must be omitted.")
        if self.type is None and "type" not in truncated:
            raise ValueError("Task topology type may be absent only when explicitly truncated.")
        return self


class TaskTopologyQuery(BaseModel):
    """Batched task links for explicitly expanded session and task branches."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    linked_session_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    session_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    expanded_parent_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    session_task_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    child_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )

    @field_validator("linked_session_ids", "expanded_parent_ids", mode="before")
    @classmethod
    def copy_branch_ids(cls, value, info) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError(f"{info.field_name} must be a sequence of strings.")
        branch_limit = (
            TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS
            if info.field_name == "linked_session_ids"
            else TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
        )
        try:
            values = islice(iter(value), branch_limit + 1)
        except TypeError as exc:
            raise ValueError(f"{info.field_name} must be a sequence of strings.") from exc
        copied: list[str] = []
        for index, item in enumerate(values):
            if index == branch_limit:
                raise ValueError(f"{info.field_name} exceeds its task topology branch limit.")
            if type(item) is not str:
                raise ValueError(f"{info.field_name} must contain only strings.")
            item = _bounded_task_topology_text(
                item,
                f"{info.field_name}[{index}]",
                max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            if item in copied:
                raise ValueError(f"{info.field_name} must not contain duplicates.")
            copied.append(item)
        return tuple(copied)

    @field_validator("session_cursors", "child_cursors", mode="before")
    @classmethod
    def copy_cursors(cls, value, info) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise ValueError(f"{info.field_name} must be an object.")
        cursor_limit = (
            TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS
            if info.field_name == "session_cursors"
            else TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
        )
        if len(value) > cursor_limit:
            raise ValueError(f"{info.field_name} exceeds its task topology branch limit.")
        copied: dict[str, str] = {}
        for raw_parent_id, raw_cursor in value.items():
            if type(raw_parent_id) is not str or type(raw_cursor) is not str:
                raise ValueError(f"{info.field_name} must map strings to strings.")
            parent_id = _bounded_task_topology_text(
                raw_parent_id,
                f"{info.field_name} key",
                max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            copied[parent_id] = _bounded_task_topology_text(
                raw_cursor,
                f"{info.field_name}[{parent_id!r}]",
                max_bytes=TASK_TOPOLOGY_MAX_CURSOR_BYTES,
            )
        return copied

    @model_validator(mode="after")
    def validate_cursor_authority(self) -> TaskTopologyQuery:
        if set(self.session_cursors).difference(self.linked_session_ids):
            raise ValueError("session_cursors keys must also appear in linked_session_ids.")
        if set(self.child_cursors).difference(self.expanded_parent_ids):
            raise ValueError("child_cursors keys must also appear in expanded_parent_ids.")
        return self


def _allocate_task_topology_branch_limits(
    query: TaskTopologyQuery,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reserve the shared node budget before stores hydrate branch candidates.

    Every requested branch receives at least one return slot. Earlier branches
    receive their requested limit while capacity permits; later branches retain
    a slot and therefore always have a truthful continuation boundary. Each
    store reads one additional sentinel row per branch to determine ``has_more``.
    """

    if type(query) is not TaskTopologyQuery:
        raise TypeError("Task topology branch allocation requires a TaskTopologyQuery.")
    requested_limits = (
        *(query.session_task_limit for _ in query.linked_session_ids),
        *(query.child_limit for _ in query.expanded_parent_ids),
    )
    if not requested_limits:
        return (), ()

    remaining = TASK_TOPOLOGY_MAX_NODES - len(query.expanded_parent_ids)
    allocated: list[int] = []
    for index, requested_limit in enumerate(requested_limits):
        remaining_branches = len(requested_limits) - index - 1
        branch_limit = min(requested_limit, remaining - remaining_branches)
        if branch_limit < 1:
            raise RuntimeError("Task topology node allocation cannot retain every branch.")
        allocated.append(branch_limit)
        remaining -= branch_limit

    session_count = len(query.linked_session_ids)
    return tuple(allocated[:session_count]), tuple(allocated[session_count:])


class TaskTopologySessionBranch(BaseModel):
    """Tasks attached to one explicitly expanded session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tasks: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "session_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologySessionBranch:
        if any(task.session_id != self.session_id for task in self.tasks):
            raise ValueError("A session-task branch contains a contradictory session link.")
        _validate_task_topology_page(
            self.tasks,
            self.next_cursor,
            self.has_more,
            scope_id=self.session_id,
            scope_kind="session",
        )
        return self


class TaskTopologyChildBranch(BaseModel):
    """Direct task children of one explicitly expanded task."""

    model_config = ConfigDict(extra="forbid")

    parent_task_id: str
    children: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("parent_task_id")
    @classmethod
    def validate_parent_task_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "parent_task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologyChildBranch:
        if any(task.parent_task_id != self.parent_task_id for task in self.children):
            raise ValueError("A child-task branch contains a contradictory parent link.")
        _validate_task_topology_page(
            self.children,
            self.next_cursor,
            self.has_more,
            scope_id=self.parent_task_id,
            scope_kind="parent_task",
        )
        return self


class TaskTopologyStoreResult(BaseModel):
    """Backend-neutral bounded task projection captured by one task-store snapshot."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    session_branches: tuple[TaskTopologySessionBranch, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    expanded_parents: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_branches: tuple[TaskTopologyChildBranch, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologyStoreResult:
        session_ids = [branch.session_id for branch in self.session_branches]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Task topology session branches must not contain duplicates.")
        expanded_ids = [node.id for node in self.expanded_parents]
        if len(set(expanded_ids)) != len(expanded_ids):
            raise ValueError("Task topology expanded parents must not contain duplicates.")
        if len(self.child_branches) != len(self.expanded_parents):
            raise ValueError("Every expanded task parent requires exactly one child branch.")
        if [branch.parent_task_id for branch in self.child_branches] != expanded_ids:
            raise ValueError("Task child branches must preserve expanded-parent order.")

        nodes = (
            *self.expanded_parents,
            *(task for branch in self.session_branches for task in branch.tasks),
            *(task for branch in self.child_branches for task in branch.children),
        )
        nodes_by_id: dict[str, TaskTopologyNode] = {}
        for node in nodes:
            prior = nodes_by_id.setdefault(node.id, node)
            if prior != node:
                raise TaskTopologyInconsistent(
                    "Task topology contains contradictory representations of one task."
                )
        if len(nodes_by_id) > TASK_TOPOLOGY_MAX_NODES:
            raise ValueError(
                f"Task topology cannot retain more than {TASK_TOPOLOGY_MAX_NODES} nodes."
            )
        _reject_loaded_task_topology_cycles(nodes_by_id)
        return self

    def validate_for_query(self, query: TaskTopologyQuery) -> TaskTopologyStoreResult:
        """Verify that a custom store honored the exact requested branches and bounds."""

        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology result validation requires a TaskTopologyQuery.")
        if tuple(branch.session_id for branch in self.session_branches) != (
            query.linked_session_ids
        ):
            raise TaskTopologyInconsistent(
                "Task topology session branches do not match the requested sessions."
            )
        if tuple(node.id for node in self.expanded_parents) != query.expanded_parent_ids:
            raise TaskTopologyInconsistent(
                "Task topology parents do not match the requested expansions."
            )
        for branch in self.session_branches:
            if len(branch.tasks) > query.session_task_limit:
                raise TaskTopologyInconsistent(
                    "A task topology session branch exceeds its requested limit."
                )
            cursor = query.session_cursors.get(branch.session_id)
            if cursor is not None and branch.tasks:
                boundary = decode_task_topology_cursor(
                    cursor,
                    scope_kind="session",
                    scope_id=branch.session_id,
                )
                if (branch.tasks[0].created_at, branch.tasks[0].id) <= boundary:
                    raise TaskTopologyInconsistent(
                        "A task topology session branch did not advance past its cursor."
                    )
        for branch in self.child_branches:
            if len(branch.children) > query.child_limit:
                raise TaskTopologyInconsistent(
                    "A task topology child branch exceeds its requested limit."
                )
            cursor = query.child_cursors.get(branch.parent_task_id)
            if cursor is not None and branch.children:
                boundary = decode_task_topology_cursor(
                    cursor,
                    scope_kind="parent_task",
                    scope_id=branch.parent_task_id,
                )
                if (branch.children[0].created_at, branch.children[0].id) <= boundary:
                    raise TaskTopologyInconsistent(
                        "A task topology child branch did not advance past its cursor."
                    )
        return self


class TaskStore(ABC):
    """Persistent store for durable work items."""

    supports_delayed_availability: ClassVar[bool] = False
    supports_task_topology: ClassVar[bool] = False
    supports_idempotent_terminalization: ClassVar[bool] = False
    supports_task_retry_series: ClassVar[bool] = False
    supports_verified_work_contracts: ClassVar[bool] = False
    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = False
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.UNVERIFIED

    # ``supports_verified_work_contracts`` alone is not settlement authority.
    # A class may set this flag to exactly ``True`` only when each verified-work
    # mutation implementation it owns has stopped mutating before its awaitable
    # returns or raises, including after caller cancellation. A subclass may
    # inherit proof for an unchanged method, but an override becomes a new
    # implementation owner and must declare its own proof.

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        """Publish one immutable version or replay its exact canonical content."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        """Load the exact contract named by ``reference`` or reject a fingerprint conflict."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        """Load a contracted task whose binding retains authority over a session.

        A terminal task does not implicitly release pending session work into the
        ordinary runtime. Until a verifier-aware release operation exists, the
        durable session binding remains authoritative and callers must start a new
        ordinary session.
        """
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        """Atomically admit a session to the ordinary, non-verifier runtime.

        Supporting stores must reject admission while a contracted task binding
        retains authority over the session and must durably prevent later contract
        attachment to an admitted session. Task terminalization alone is not a
        release. Repeated admission of the same session is idempotent.
        """
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        contract: WorkContractRef,
    ) -> Task:
        """Claim-fence an unsupported contracted task into operator attention."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        """Create or replay one bounded execution attempt under a task's frozen contract."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        """Load one work attempt by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        """Persist a worker proposal without granting completion authority."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        """Load one completion proposal by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        """Claim bounded exclusive authority to verify one undecided proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        """Load the latest verification claim, including an expired claim for recovery."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        """Extend one exact live verification claim without changing its owner."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        """Persist the one authoritative verifier decision for a proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision(self, decision_id: str) -> CompletionDecision | None:
        """Load one completion decision by stable identity."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        """Load the one authoritative decision published for a proposal."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        """Apply or exactly replay a decision-bound task transition."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        """Load exact durable evidence for decision application reconciliation."""
        raise NotImplementedError("This TaskStore does not support verified work contracts.")

    @abstractmethod
    async def create_task(self, request: TaskCreate) -> Task:
        """Create a task and its immutable invocation provenance atomically.

        Implementations must mint the final task ID, load any requested parent,
        and call ``task_invocation_for_create`` inside the same create boundary.
        """

    @abstractmethod
    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        """Atomically create a running task already attached to its session.

        ``request.session_id`` is required. This avoids leaving an attached,
        unclaimable pending task if a process stops between separate create and
        start operations. ``session_invocation`` must describe that exact
        session so the atomic insert cannot create a contradictory structural
        attachment and invocation record.
        """

    @abstractmethod
    async def load_task(self, task_id: str) -> Task | None:
        """Load a task by id."""

    @abstractmethod
    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        """Load bounded immutable provenance without task payloads or metadata."""

    @abstractmethod
    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        """List tasks for dashboards, queues, and orchestration."""

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        """Count current task states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable when they do not expose this control-plane read model.
        """
        raise NotImplementedError(
            "This TaskStore does not support operational aggregate snapshots."
        )

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        """Read bounded task/session and direct-child branches.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable while advertising the operation as unsupported.
        """
        raise NotImplementedError("This TaskStore does not support topology queries.")

    @abstractmethod
    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        """Mark a pending task as running, optionally attached to a session.

        A session provenance binding is required for attachment. Stores must
        validate it before changing lifecycle state.
        """

    @abstractmethod
    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        session_invocation: SessionInvocationBinding,
        worker_id: str,
    ) -> Task:
        """Attach a live worker-claimed task to a session and mark it running.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as completed.

        If ``worker_id`` is given, the update raises ``TaskClaimLost`` unless that
        worker still owns an active lease on the task, so a worker that lost its
        lease cannot clobber a task another worker has since reclaimed.
        """

    @abstractmethod
    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as failed.

        If ``worker_id`` is given, the update raises ``TaskClaimLost`` unless that
        worker still owns an active lease on the task.
        """

    async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
        """Atomically terminalize one live claim or replay its exact receipt.

        Custom stores opt into this operation by overriding it. The default keeps
        existing out-of-tree ``TaskStore`` implementations instantiable without
        claiming acknowledgement-loss safety they do not provide.
        """
        raise NotImplementedError(
            "This TaskStore does not support idempotent task terminalization."
        )

    async def load_task_terminalization_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskTerminalizationReceipt | None:
        """Load exact durable commit evidence for receipt reconciliation."""
        raise NotImplementedError("This TaskStore does not support task terminalization receipts.")

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        """Atomically record one attempt and optionally create its delayed successor."""

        raise NotImplementedError("This TaskStore does not support task retry series.")

    async def load_task_retry_settlement(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskRetrySettlementResult | None:
        """Load an exact retry-attempt settlement receipt for acknowledgement recovery."""

        raise NotImplementedError("This TaskStore does not support task retry series.")

    async def enforce_task_retry_deadline(
        self,
        task_id: str,
        worker_id: str,
        *,
        token_count: int = 0,
        estimated_cost: Decimal = Decimal(0),
    ) -> TaskRetrySettlementResult | None:
        """Atomically terminalize an owned attempt only when store time exhausted it.

        Returning ``None`` is positive store evidence that the cumulative deadline
        had not elapsed at this check. Retry-capable custom stores must override
        this operation so workers never substitute their own wall clock.
        """

        raise NotImplementedError("This TaskStore does not support task retry deadlines.")

    async def task_retry_deadline_elapsed(
        self,
        task_id: str,
        worker_id: str,
    ) -> bool:
        """Return store-authoritative elapsed evidence without releasing ownership."""

        raise NotImplementedError("This TaskStore does not support task retry deadlines.")

    @abstractmethod
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        """Cancel idle work or durably request cancellation from its live owner.

        Retry-series attempts with a live worker remain fenced until that worker
        proves its dispatched work quiescent and commits the cancellation receipt.
        """

    @abstractmethod
    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Pause a pending or unattached running task until app code resumes it."""

    @abstractmethod
    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as blocked on an external dependency."""

    @abstractmethod
    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as waiting for human/operator input."""

    @abstractmethod
    async def resume_task(self, task_id: str) -> Task:
        """Return a paused, blocked, or attention-needed task to the pending queue."""

    @abstractmethod
    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        """Atomically claim the next pending task matching ``query``."""

    @abstractmethod
    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        """Extend a worker-owned active lease.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def release_task(self, task_id: str, worker_id: str) -> Task:
        """Release a claimed task back to pending and clear worker ownership.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        """Release worker ownership while preserving a running task's session link.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        """Return expired claimed task leases to pending."""


class InMemoryTaskStore(TaskStore):
    """In-process task store for tests, local development, and examples."""

    supports_delayed_availability: ClassVar[bool] = True
    supports_task_topology: ClassVar[bool] = True
    supports_idempotent_terminalization: ClassVar[bool] = True
    supports_task_retry_series: ClassVar[bool] = True
    supports_verified_work_contracts: ClassVar[bool] = True
    verified_work_mutations_are_cancellation_quiescent: ClassVar[bool] = True
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DEVELOPMENT

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._clock = utc_clock(clock)
        self._tasks: dict[str, Task] = {}
        self._terminalization_receipts: dict[tuple[str, str], TaskTerminalizationReceipt] = {}
        self._retry_settlements: dict[tuple[str, str], TaskRetrySettlementResult] = {}
        self._work_contracts: dict[tuple[str, int], WorkContract] = {}
        self._work_attempts: dict[str, WorkAttempt] = {}
        self._attempt_ids_by_task: dict[str, list[str]] = {}
        self._completion_proposals: dict[str, CompletionProposal] = {}
        self._proposal_id_by_attempt: dict[str, str] = {}
        self._completion_verification_claims: dict[str, CompletionVerificationClaim] = {}
        self._verification_claims_by_id: dict[str, CompletionVerificationClaim] = {}
        self._completion_decisions: dict[str, CompletionDecision] = {}
        self._decision_id_by_proposal: dict[str, str] = {}
        self._decision_application_receipts: dict[
            tuple[str, str], CompletionDecisionApplicationReceipt
        ] = {}
        self._decision_application_key_by_decision: dict[str, tuple[str, str]] = {}
        self._ordinary_execution_session_ids: set[str] = set()
        self._contracted_task_ids_by_session: dict[str, dict[str, None]] = {}
        self._task_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._task_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}

    async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
        contract = copy_work_contract(contract)
        key = (contract.contract_id, contract.version)
        async with self._lock:
            existing = self._work_contracts.get(key)
            if existing is not None:
                if existing != contract:
                    raise WorkContractConflict(
                        "Work-contract identity is already bound to different content."
                    )
                return copy_work_contract(existing)
            if contract.supersedes is not None:
                predecessor = self._work_contracts.get(
                    (contract.supersedes.contract_id, contract.supersedes.version)
                )
                if predecessor is None:
                    raise WorkContractConflict("Work-contract predecessor has not been published.")
                if predecessor.reference() != contract.supersedes:
                    raise WorkContractConflict(
                        "Work-contract predecessor fingerprint conflicts with durable history."
                    )
            self._work_contracts[key] = contract
            return copy_work_contract(contract)

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        copied_reference = copy_work_contract_ref(reference)
        if copied_reference is None:  # pragma: no cover - excluded by the public type
            raise TypeError("reference must be a WorkContractRef.")
        async with self._lock:
            contract = self._work_contracts.get(
                (copied_reference.contract_id, copied_reference.version)
            )
            if contract is None:
                return None
            if contract.fingerprint != copied_reference.fingerprint:
                raise WorkContractConflict(
                    "Work-contract reference conflicts with the published fingerprint."
                )
            return copy_work_contract(contract)

    async def load_active_work_contract_task_for_session(
        self,
        session_id: str,
    ) -> Task | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            task = self._active_work_contract_task_for_session(session_id)
            return None if task is None else task.model_copy(deep=True)

    async def admit_ordinary_session_execution(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            contracted_task = self._active_work_contract_task_for_session(session_id)
            requires_completion_decision = contracted_task is not None
            del contracted_task
            if requires_completion_decision:
                raise TaskCompletionDecisionRequired(
                    "Contracted tasks require the verifier-aware execution entrance."
                ) from None
            self._ordinary_execution_session_ids.add(session_id)

    async def hold_claimed_work_contract_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        contract: WorkContractRef,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        copied_contract = copy_work_contract_ref(contract)
        if copied_contract is None:  # pragma: no cover - excluded by the public type
            raise TypeError("contract must be a WorkContractRef.")
        async with self._lock:
            task = self._require_task(task_id)
            now = datetime.now(UTC)
            _ensure_owned_active_task_lease(task, worker_id, now=now)
            if task.status is not TaskStatus.CLAIMED or task.session_id is not None:
                raise TaskClaimLost("Only the current worker may park its unattached claimed task.")
            self._ensure_task_contract_matches(task, copied_contract)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.NEEDS_ATTENTION,
                    "status_reason": "verified_work_contract_runner_required",
                    "status_payload": {
                        "contract_id": copied_contract.contract_id,
                        "contract_version": copied_contract.version,
                    },
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def begin_work_attempt(self, request: WorkAttemptCreate) -> WorkAttempt:
        request = copy_work_attempt_create(request)
        request_sha256 = work_attempt_request_sha256(request)
        async with self._lock:
            existing = self._work_attempts.get(request.attempt_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Work-attempt identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            task = self._require_task(request.task_id)
            contract = self._ensure_task_contract_matches(task, request.contract)
            if task.status is not TaskStatus.RUNNING:
                raise ValueError("Work attempts require a running contracted task.")
            if task.session_id != request.session_id:
                raise WorkCompletionConflict("Work attempt is bound to a different task session.")
            self._ensure_attempt_worker_matches(task, request.worker_id)
            attempt_ids = self._attempt_ids_by_task.get(task.id, [])
            if len(attempt_ids) >= contract.continuation_policy.max_attempts:
                raise WorkCompletionConflict(
                    "Work-contract attempt limit forbids another work attempt."
                )
            if attempt_ids:
                prior_attempt_id = attempt_ids[-1]
                prior_proposal_id = self._proposal_id_by_attempt.get(prior_attempt_id)
                prior_decision_id = (
                    None
                    if prior_proposal_id is None
                    else self._decision_id_by_proposal.get(prior_proposal_id)
                )
                if prior_decision_id is None:
                    raise WorkCompletionConflict(
                        "A prior work attempt has not reached a durable decision."
                    )
                if prior_decision_id not in self._decision_application_key_by_decision:
                    raise WorkCompletionConflict(
                        "A prior verifier decision has not reached durable task application."
                    )
            attempt = WorkAttempt(
                attempt_id=request.attempt_id,
                task_id=request.task_id,
                session_id=request.session_id,
                contract=request.contract,
                execution_profile_fingerprint=request.execution_profile_fingerprint,
                worker_id=request.worker_id,
                ordinal=len(attempt_ids) + 1,
                request_sha256=request_sha256,
                started_at=self._clock(),
            )
            self._work_attempts[attempt.attempt_id] = attempt
            self._attempt_ids_by_task.setdefault(task.id, []).append(attempt.attempt_id)
            return attempt.model_copy(deep=True)

    async def load_work_attempt(self, attempt_id: str) -> WorkAttempt | None:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        async with self._lock:
            attempt = self._work_attempts.get(attempt_id)
            return None if attempt is None else attempt.model_copy(deep=True)

    async def submit_completion_proposal(
        self,
        request: CompletionProposalCreate,
    ) -> CompletionProposal:
        request = copy_completion_proposal_create(request)
        request_sha256 = completion_proposal_request_sha256(request)
        async with self._lock:
            existing = self._completion_proposals.get(request.proposal_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-proposal identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            prior_proposal_id = self._proposal_id_by_attempt.get(request.attempt_id)
            if prior_proposal_id is not None:
                raise WorkCompletionConflict(
                    "Work attempt already has a different completion proposal."
                )
            attempt = self._require_work_attempt(request.attempt_id)
            task = self._require_task(attempt.task_id)
            self._ensure_attempt_is_current(task, attempt)
            proposal = CompletionProposal(
                proposal_id=request.proposal_id,
                attempt_id=request.attempt_id,
                result=request.result,
                evidence_references=request.evidence_references,
                task_id=attempt.task_id,
                contract=attempt.contract,
                request_sha256=request_sha256,
                proposed_at=self._clock(),
            )
            self._completion_proposals[proposal.proposal_id] = proposal
            self._proposal_id_by_attempt[attempt.attempt_id] = proposal.proposal_id
            return proposal.model_copy(deep=True)

    async def load_completion_proposal(self, proposal_id: str) -> CompletionProposal | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            proposal = self._completion_proposals.get(proposal_id)
            return None if proposal is None else proposal.model_copy(deep=True)

    async def claim_completion_verification(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            claim_by_id = self._verification_claims_by_id.get(request.claim_id)
            if claim_by_id is not None and (
                claim_by_id.proposal_id != request.proposal_id
                or claim_by_id.request_sha256 != request_sha256
            ):
                raise WorkCompletionConflict(
                    "Verification-claim identity is already bound to another request."
                )
            proposal = self._require_completion_proposal(request.proposal_id)
            contract = self._require_work_contract(proposal.contract)
            if request.verifier != contract.verifier:
                raise WorkCompletionConflict(
                    "Verification claim uses a verifier other than the frozen contract verifier."
                )
            now = self._clock()
            current = self._completion_verification_claims.get(request.proposal_id)
            if (
                current is not None
                and current.claim_id == request.claim_id
                and current.request_sha256 == request_sha256
            ):
                if (
                    current.lease_expires_at > now
                    or proposal.proposal_id in self._decision_id_by_proposal
                ):
                    return current.model_copy(deep=True)
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            if proposal.proposal_id in self._decision_id_by_proposal:
                raise WorkCompletionConflict("Completion proposal already has a durable decision.")
            if current is not None and current.lease_expires_at > now:
                raise CompletionVerificationClaimLost(
                    "Completion proposal is owned by another live verifier claim."
                )
            if claim_by_id is not None:
                raise CompletionVerificationClaimLost(
                    "Verification claim expired and cannot regain authority by replay."
                )
            self._ensure_completion_proposal_is_current(proposal)
            attempt_number = 1 if current is None else current.attempt_number + 1
            claim = CompletionVerificationClaim(
                claim_id=request.claim_id,
                proposal_id=request.proposal_id,
                worker_id=request.worker_id,
                execution_owner_id=request.execution_owner_id,
                execution_timeout_seconds=request.execution_timeout_seconds,
                verifier=request.verifier,
                attempt_number=attempt_number,
                request_sha256=request_sha256,
                claimed_at=now,
                lease_expires_at=now + timedelta(seconds=request.lease_seconds),
            )
            self._completion_verification_claims[proposal.proposal_id] = claim
            self._verification_claims_by_id[claim.claim_id] = claim
            return claim.model_copy(deep=True)

    async def load_completion_verification_claim(
        self,
        proposal_id: str,
    ) -> CompletionVerificationClaim | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            claim = self._completion_verification_claims.get(proposal_id)
            return None if claim is None else claim.model_copy(deep=True)

    async def renew_completion_verification_claim(
        self,
        request: CompletionVerificationClaimRequest,
    ) -> CompletionVerificationClaim:
        request = copy_completion_verification_claim_request(request)
        request_sha256 = completion_verification_claim_request_sha256(request)
        async with self._lock:
            proposal = self._require_completion_proposal(request.proposal_id)
            current = self._completion_verification_claims.get(request.proposal_id)
            now = self._clock()
            if (
                current is None
                or current.claim_id != request.claim_id
                or current.proposal_id != request.proposal_id
                or current.worker_id != request.worker_id
                or current.execution_owner_id != request.execution_owner_id
                or current.execution_timeout_seconds != request.execution_timeout_seconds
                or current.verifier != request.verifier
                or current.request_sha256 != request_sha256
                or current.lease_expires_at <= now
                or proposal.proposal_id in self._decision_id_by_proposal
            ):
                raise CompletionVerificationClaimLost(
                    "Verification claim cannot be renewed without exact current live authority."
                )
            self._ensure_completion_proposal_is_current(proposal)
            renewed = CompletionVerificationClaim(
                claim_id=current.claim_id,
                proposal_id=current.proposal_id,
                worker_id=current.worker_id,
                execution_owner_id=current.execution_owner_id,
                execution_timeout_seconds=current.execution_timeout_seconds,
                verifier=current.verifier,
                attempt_number=current.attempt_number,
                request_sha256=current.request_sha256,
                claimed_at=current.claimed_at,
                lease_expires_at=max(
                    current.lease_expires_at,
                    now + timedelta(seconds=request.lease_seconds),
                ),
            )
            self._completion_verification_claims[proposal.proposal_id] = renewed
            self._verification_claims_by_id[renewed.claim_id] = renewed
            return renewed.model_copy(deep=True)

    async def record_completion_decision(
        self,
        request: CompletionDecisionCreate,
    ) -> CompletionDecision:
        request = copy_completion_decision_create(request)
        request_sha256 = completion_decision_request_sha256(request)
        async with self._lock:
            existing = self._completion_decisions.get(request.decision_id)
            if existing is not None:
                if existing.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Completion-decision identity is already bound to another request."
                    )
                return existing.model_copy(deep=True)
            prior_decision_id = self._decision_id_by_proposal.get(request.proposal_id)
            if prior_decision_id is not None:
                raise WorkCompletionConflict(
                    "Completion proposal already has a different durable decision."
                )
            proposal = self._require_completion_proposal(request.proposal_id)
            claim = self._completion_verification_claims.get(proposal.proposal_id)
            now = self._clock()
            if (
                claim is None
                or claim.claim_id != request.claim_id
                or claim.worker_id != request.worker_id
                or claim.verifier != request.verifier
                or claim.lease_expires_at <= now
            ):
                raise CompletionVerificationClaimLost(
                    "Completion decision requires the current live verifier claim."
                )
            self._ensure_completion_proposal_is_current(proposal)
            contract = self._require_work_contract(proposal.contract)
            validate_completion_decision_contract(contract, request)
            decision = CompletionDecision(
                decision_id=request.decision_id,
                proposal_id=request.proposal_id,
                claim_id=request.claim_id,
                worker_id=request.worker_id,
                verifier=request.verifier,
                decision_version=request.decision_version,
                verdict=request.verdict,
                criterion_outcomes=request.criterion_outcomes,
                constraint_outcomes=request.constraint_outcomes,
                gaps=request.gaps,
                evidence_references=request.evidence_references,
                task_id=proposal.task_id,
                attempt_id=proposal.attempt_id,
                contract=proposal.contract,
                request_sha256=request_sha256,
                gap_fingerprint=completion_gap_fingerprint(request),
                decided_at=now,
            )
            self._completion_decisions[decision.decision_id] = decision
            self._decision_id_by_proposal[proposal.proposal_id] = decision.decision_id
            return decision.model_copy(deep=True)

    async def load_completion_decision(self, decision_id: str) -> CompletionDecision | None:
        decision_id = require_clean_nonblank(decision_id, "decision_id")
        async with self._lock:
            decision = self._completion_decisions.get(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def load_completion_decision_for_proposal(
        self,
        proposal_id: str,
    ) -> CompletionDecision | None:
        proposal_id = require_clean_nonblank(proposal_id, "proposal_id")
        async with self._lock:
            decision_id = self._decision_id_by_proposal.get(proposal_id)
            decision = None if decision_id is None else self._completion_decisions.get(decision_id)
            return None if decision is None else decision.model_copy(deep=True)

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        try:
            copied_request = copy_completion_decision_application_request(request)
        except BaseException:
            del request
            raise
        request = copied_request
        del copied_request
        request_sha256 = completion_decision_application_request_sha256(request)
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            receipt = self._decision_application_receipts.get(receipt_key)
            if receipt is not None:
                if receipt.request_sha256 != request_sha256:
                    raise WorkCompletionConflict(
                        "Decision-application identity is already bound to another request."
                    )
                return receipt.task.model_copy(deep=True)
            prior_receipt_key = self._decision_application_key_by_decision.get(request.decision_id)
            if prior_receipt_key is not None:
                raise WorkCompletionConflict(
                    "Completion decision was already applied under another identity."
                )
            task = self._require_task(request.task_id)
            decision = self._completion_decisions.get(request.decision_id)
            if decision is None:
                raise KeyError(f"Completion decision not found: {request.decision_id}")
            if decision.task_id != task.id:
                raise WorkCompletionConflict("Completion decision belongs to another task.")
            contract = self._ensure_task_contract_matches(task, decision.contract)
            attempt = self._require_work_attempt(decision.attempt_id)
            # The durable verifier decision, exact application tuple, and latest
            # attempt identity authorize this transition. The originating task
            # worker may have crashed or its lease may have expired while an
            # independent verifier was running.
            self._ensure_decision_attempt_is_current(task, attempt)
            proposal = self._require_completion_proposal(decision.proposal_id)
            applied_at = _task_lifecycle_now(task)
            task_changed = False
            if decision.verdict is CompletionVerdict.ACCEPTED:
                if request.result is None or request.result_reference is None:
                    raise ValueError(
                        "Accepted completion decisions require a verified task result."
                    )
                if request.result_reference != proposal.result:
                    raise WorkCompletionConflict(
                        "Decision application result conflicts with the accepted proposal."
                    )
                task = self._prepare_finished_task(
                    task.id,
                    TaskStatus.COMPLETED,
                    result=request.result,
                    error=None,
                    worker_id=None,
                    accepted_decision_id=decision.decision_id,
                    now=applied_at,
                )
                task_changed = True
            else:
                if request.result is not None:
                    raise ValueError("Non-accepted completion decisions cannot carry a result.")
                if decision.verdict is CompletionVerdict.BLOCKED:
                    task = self._prepare_held_task_from_completion_decision(
                        task,
                        TaskStatus.BLOCKED,
                        decision=decision,
                        now=applied_at,
                    )
                    task_changed = True
                elif decision.verdict is CompletionVerdict.NEEDS_REVIEW:
                    task = self._prepare_held_task_from_completion_decision(
                        task,
                        TaskStatus.NEEDS_ATTENTION,
                        decision=decision,
                        now=applied_at,
                    )
                    task_changed = True
                elif decision.verdict is CompletionVerdict.REJECTED:
                    rejection_hold = self._completion_rejection_hold(
                        contract,
                        decision,
                        attempt,
                    )
                    if rejection_hold is not None:
                        hold_status, status_reason = rejection_hold
                        task = self._prepare_held_task_from_completion_decision(
                            task,
                            hold_status,
                            decision=decision,
                            now=applied_at,
                            status_reason=status_reason,
                        )
                        task_changed = True
                    elif task.worker_id is not None or task.lease_expires_at is not None:
                        # A decision closes the attempt and fences its worker. A
                        # verifier-aware owner can begin the next attempt without
                        # inheriting authority from the prior attempt's lease.
                        task = task.model_copy(
                            update={
                                "worker_id": None,
                                "lease_expires_at": None,
                                "updated_at": applied_at,
                            }
                        )
                        task_changed = True
            receipt = CompletionDecisionApplicationReceipt(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key=request.idempotency_key,
                request_sha256=request_sha256,
                task=task,
                applied_at=applied_at,
            )
            if task_changed:
                self._store_task(task)
            self._decision_application_receipts[receipt_key] = receipt
            self._decision_application_key_by_decision[decision.decision_id] = receipt_key
            return task.model_copy(deep=True)

    async def load_completion_decision_application_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        idempotency_key = validate_work_completion_idempotency_key(idempotency_key)
        async with self._lock:
            receipt = self._decision_application_receipts.get((task_id, idempotency_key))
            if receipt is None:
                return None
            return CompletionDecisionApplicationReceipt.model_validate(
                receipt.model_dump(mode="python", warnings=False)
            )

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task_id = request.task_id or str(uuid4())
            parent = self._task_parent_for_create(request, task_id=task_id)
            if request.work_contract is not None:
                self._require_work_contract(request.work_contract)
                self._ensure_contract_session_accepts_attachment(
                    request.work_contract,
                    request.session_id,
                )
            task = _task_from_create(
                request,
                task_id=task_id,
                parent_task=parent,
                retry_started_at=self._clock(),
                supports_verified_work_contracts=True,
            )
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            return task.model_copy(deep=True)

    async def create_running_task(
        self,
        request: TaskCreate,
        *,
        session_invocation: SessionInvocationBinding,
    ) -> Task:
        request = copy_task_create(request)
        session_binding = _copy_required_session_binding(session_invocation)
        async with self._lock:
            task_id = request.task_id or str(uuid4())
            parent = self._task_parent_for_create(request, task_id=task_id)
            if request.work_contract is not None:
                self._require_work_contract(request.work_contract)
                self._ensure_contract_session_accepts_attachment(
                    request.work_contract,
                    request.session_id,
                )
            task = _running_task_from_create(
                request,
                task_id=task_id,
                parent_task=parent,
                session_invocation=session_binding,
                retry_started_at=self._clock(),
                supports_verified_work_contracts=True,
            )
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            return task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return task.model_copy(deep=True)

    async def load_invocation_snapshot(
        self,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return TaskInvocationSnapshot(
                id=task.id,
                session_id=task.session_id,
                invocation=task.invocation,
            )

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        async with self._lock:
            tasks = [task for task in self._tasks.values() if _task_matches(task, query)]
            tasks = _sort_tasks(tasks, query.order_by)
            page = tasks[query.offset : query.offset + query.limit]
            return [task.model_copy(deep=True) for task in page]

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        async with self._lock:
            session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(
                query
            )
            expanded_parents: list[TaskTopologyNode] = []
            for parent_id in query.expanded_parent_ids:
                parent = self._tasks.get(parent_id)
                if parent is None:
                    raise KeyError(f"Task not found: {parent_id}")
                expanded_parents.append(TaskTopologyNode.from_task(parent))

            session_candidates: list[list[TaskTopologyNode]] = []
            for session_id, branch_limit in zip(
                query.linked_session_ids,
                session_branch_limits,
                strict=True,
            ):
                session_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_session.get(session_id, ()),
                            cursor=query.session_cursors.get(session_id),
                            scope_kind="session",
                            scope_id=session_id,
                            limit=branch_limit,
                        )
                    ]
                )

            child_candidates: list[list[TaskTopologyNode]] = []
            for parent, branch_limit in zip(
                expanded_parents,
                child_branch_limits,
                strict=True,
            ):
                child_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_parent.get(parent.id, ()),
                            cursor=query.child_cursors.get(parent.id),
                            scope_kind="parent_task",
                            scope_id=parent.id,
                            limit=branch_limit,
                        )
                    ]
                )

            async def load_parent_links(task_ids: tuple[str, ...]) -> Mapping[str, str | None]:
                links: dict[str, str | None] = {}
                for task_id in task_ids:
                    task = self._tasks.get(task_id)
                    if task is None:
                        continue
                    links[task_id] = _bounded_optional_task_topology_parent_id(task.parent_task_id)
                return links

            await _validate_task_topology_ancestry(
                (
                    *expanded_parents,
                    *(task for branch in session_candidates for task in branch),
                    *(task for branch in child_candidates for task in branch),
                ),
                load_parent_links,
            )
            result = build_task_topology_result(
                observed_at=datetime.now(UTC),
                linked_session_ids=query.linked_session_ids,
                session_branch_candidates=session_candidates,
                session_branch_limits=session_branch_limits,
                expanded_parents=expanded_parents,
                child_branch_candidates=child_candidates,
                child_branch_limits=child_branch_limits,
                session_task_limit=query.session_task_limit,
                child_limit=query.child_limit,
            )
            return result

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        task_query = task_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = self._clock()
            counts = {status: 0 for status in TaskStatus}
            total_count = 0
            claimable_pending_count = 0
            scheduled_pending_count = 0
            for task in self._tasks.values():
                if _task_matches(task, task_query):
                    counts[task.status] += 1
                    total_count += 1
                    if task.status == TaskStatus.PENDING:
                        if task.available_at is not None and task.available_at > as_of:
                            scheduled_pending_count += 1
                        elif task.session_id is None:
                            claimable_pending_count += 1
            return TaskOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=TaskStatusCounts.model_validate(counts),
                claimable_pending_count=claimable_pending_count,
                scheduled_pending_count=scheduled_pending_count,
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
        session_invocation: SessionInvocationBinding | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        session_binding = _copy_optional_session_binding(session_invocation)
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            now = datetime.now(UTC)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            effective_session_id = _task_session_id_for_start(
                task_id=task.id,
                stored_session_id=task.session_id,
                requested_session_id=session_id,
            )
            self._ensure_contract_session_accepts_attachment(
                task.work_contract,
                effective_session_id,
                require_session=True,
            )
            _task_invocation_for_attachment(
                task.invocation,
                session_id=effective_session_id,
                session_binding=session_binding,
            )
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": effective_session_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        session_invocation: SessionInvocationBinding,
        worker_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        session_binding = _copy_required_session_binding(session_invocation)
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_retry_series_queue_attempt(task.retry_series)
            now = datetime.now(UTC)
            if not _can_attach_claimed_task(task, worker_id=worker_id, now=now):
                _raise_task_claim_attach_error(task, worker_id, now=now)
            self._ensure_contract_session_accepts_attachment(
                task.work_contract,
                session_id,
            )
            _task_invocation_for_attachment(
                task.invocation,
                session_id=session_id,
                session_binding=session_binding,
            )
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": session_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
            )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
            )

    async def terminalize_task(self, request: TaskTerminalizationRequest) -> Task:
        request, request_sha256 = prepare_task_terminalization(request)
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            existing = self._terminalization_receipts.get(receipt_key)
            if existing is not None:
                return _replay_task_terminalization_receipt(
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )

            task = self._require_task(request.task_id)
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                raise TaskTerminalizationConflict(
                    "Task is terminal without the matching terminalization receipt."
                )
            _ensure_owned_active_task_lease(task, request.worker_id)

            status = (
                TaskStatus.COMPLETED
                if request.kind is TaskTerminalKind.COMPLETED
                else TaskStatus.FAILED
            )
            terminal_task = self._finish_task(
                request.task_id,
                status,
                result=request.result,
                error=request.error,
                worker_id=request.worker_id,
            )
            self._terminalization_receipts[receipt_key] = TaskTerminalizationReceipt(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                worker_id=request.worker_id,
                kind=request.kind,
                request_sha256=request_sha256,
                task=terminal_task,
                committed_at=datetime.now(UTC),
            )
            return terminal_task.model_copy(deep=True)

    async def load_task_terminalization_receipt(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskTerminalizationReceipt | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        async with self._lock:
            receipt = self._terminalization_receipts.get((task_id, idempotency_key))
            if receipt is None:
                return None
            return TaskTerminalizationReceipt(
                task_id=receipt.task_id,
                idempotency_key=receipt.idempotency_key,
                worker_id=receipt.worker_id,
                kind=receipt.kind,
                request_sha256=receipt.request_sha256,
                task=receipt.task.model_copy(deep=True),
                committed_at=receipt.committed_at,
            )

    async def settle_task_retry_attempt(
        self,
        request: TaskRetrySettlementRequest,
    ) -> TaskRetrySettlementResult:
        request, request_sha256 = prepare_task_retry_settlement(request)
        receipt_key = (request.task_id, request.idempotency_key)
        async with self._lock:
            existing = self._retry_settlements.get(receipt_key)
            if existing is not None:
                return _replay_task_retry_settlement(
                    request_sha256=request_sha256,
                    receipt=existing,
                    current_task=self._tasks.get(request.task_id),
                )
            task = self._require_task(request.task_id)
            now = datetime.now(UTC)
            _ensure_owned_active_task_lease(task, request.worker_id, now=now)
            if task.retry_series is None:
                raise ValueError("Task does not belong to a retry series.")
            settled, successor = _settled_task_retry_attempt(
                task,
                request,
                now=now,
                series_now=self._clock(),
            )
            if successor is not None and successor.id in self._tasks:
                raise TaskTerminalizationConflict(
                    "Task retry successor identity is already occupied."
                )
            self._store_task(settled)
            if successor is not None:
                self._store_task(successor)
            receipt = TaskRetrySettlementResult(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                request_sha256=request_sha256,
                task=settled,
                successor=successor,
                events=_task_retry_events(settled, occurred_at=now),
                committed_at=now,
            )
            self._retry_settlements[receipt_key] = receipt
            return receipt.model_copy(deep=True)

    async def enforce_task_retry_deadline(
        self,
        task_id: str,
        worker_id: str,
        *,
        token_count: int = 0,
        estimated_cost: Decimal = Decimal(0),
    ) -> TaskRetrySettlementResult | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        token_count, estimated_cost = _validated_task_retry_terminal_accounting(
            token_count=token_count,
            estimated_cost=estimated_cost,
        )
        async with self._lock:
            task = self._require_task(task_id)
            lease_now = datetime.now(UTC)
            _ensure_owned_active_task_lease(task, worker_id, now=lease_now)
            series_now = self._clock()
            if not _claimed_task_retry_attempt_elapsed(task, series_now=series_now):
                return None
            receipt = _elapsed_claimed_task_retry_settlement(
                task,
                committed_at=lease_now,
                token_count=token_count,
                estimated_cost=estimated_cost,
            )
            self._store_task(receipt.task)
            self._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
            return receipt.model_copy(deep=True)

    async def task_retry_deadline_elapsed(
        self,
        task_id: str,
        worker_id: str,
    ) -> bool:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_owned_active_task_lease(task, worker_id, now=datetime.now(UTC))
            return _claimed_task_retry_attempt_elapsed(task, series_now=self._clock())

    async def load_task_retry_settlement(
        self,
        task_id: str,
        idempotency_key: str,
    ) -> TaskRetrySettlementResult | None:
        task_id, idempotency_key = prepare_task_terminalization_receipt_lookup(
            task_id,
            idempotency_key,
        )
        async with self._lock:
            receipt = self._retry_settlements.get((task_id, idempotency_key))
            return None if receipt is None else receipt.model_copy(deep=True)

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.PAUSED,
            reason=reason,
            payload=payload,
        )

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.BLOCKED,
            reason=reason,
            payload=payload,
        )

    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.NEEDS_ATTENTION,
            reason=reason,
            payload=payload,
        )

    async def resume_task(self, task_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_can_resume_task(task)
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "status_reason": None,
                    "status_payload": None,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        async with self._lock:
            availability_now = self._clock()
            now = datetime.now(UTC)
            for task in tuple(self._tasks.values()):
                if _task_retry_attempt_elapsed(task, series_now=availability_now):
                    receipt = _expired_task_retry_settlement(
                        task,
                        committed_at=now,
                        series_now=availability_now,
                    )
                    self._store_task(receipt.task)
                    self._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
            candidates = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.PENDING
                and task.session_id is None
                and (task.available_at is None or task.available_at <= availability_now)
                and not _task_retry_attempt_elapsed(task, series_now=availability_now)
                and _task_matches_claim_filter(task, query)
            ]
            if not candidates:
                return None
            # Claiming is always FIFO by creation time, independent of the query's
            # display ordering, so the oldest pending task is dispatched first.
            task = _sort_tasks(candidates, TaskOrder.CREATED_AT_ASC)[0]
            if task.work_contract is not None:
                validate_work_completion_linked_id(worker_id, "worker_id")
            updated = task.model_copy(
                update={
                    "status": TaskStatus.CLAIMED,
                    "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        extend_seconds = _validate_positive_int(extend_seconds, "extend_seconds")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "lease_expires_at": now + timedelta(seconds=extend_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def release_task(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            if task.session_id is not None:
                raise ValueError(
                    f"Task {task.id} is already attached to session {task.session_id}."
                )
            if task.status is not TaskStatus.CLAIMED:
                raise ValueError(f"Task {task.id} is not claimed.")
            if _task_retry_cancellation_requested(task):
                raise TaskTerminalizationConflict(
                    "Task retry cancellation is still draining under its current owner."
                )
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            if task.status is not TaskStatus.RUNNING:
                raise ValueError(f"Task {task.id} is not running.")
            if task.session_id is None:
                raise ValueError(f"Task {task.id} is not attached to a session.")
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        async with self._lock:
            now = datetime.now(UTC)
            expired = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and not _task_retry_cancellation_requested(task)
                and task.lease_expires_at is not None
                and task.lease_expires_at <= now
                and _task_matches_claim_filter(task, query)
            ]
            expired = _sort_tasks(expired, TaskOrder.UPDATED_AT_ASC)
            reclaimed: list[Task] = []
            for task in expired[:max_reclaims]:
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.PENDING,
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                )
                self._store_task(updated)
                reclaimed.append(updated.model_copy(deep=True))
            return reclaimed

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _require_work_contract(self, reference: WorkContractRef) -> WorkContract:
        contract = self._work_contracts.get((reference.contract_id, reference.version))
        if contract is None:
            raise WorkContractConflict("Referenced work contract has not been published.")
        if contract.fingerprint != reference.fingerprint:
            raise WorkContractConflict(
                "Work-contract reference conflicts with the published fingerprint."
            )
        return contract

    def _ensure_task_contract_matches(
        self,
        task: Task,
        reference: WorkContractRef,
    ) -> WorkContract:
        contract = self._require_work_contract(reference)
        if task.work_contract is None:
            raise WorkCompletionConflict("Task is not bound to a work contract.")
        if task.work_contract != reference:
            raise WorkCompletionConflict(
                "Work operation conflicts with the task's frozen contract binding."
            )
        return contract

    def _require_work_attempt(self, attempt_id: str) -> WorkAttempt:
        attempt = self._work_attempts.get(attempt_id)
        if attempt is None:
            raise KeyError(f"Work attempt not found: {attempt_id}")
        return attempt

    def _require_completion_proposal(self, proposal_id: str) -> CompletionProposal:
        proposal = self._completion_proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Completion proposal not found: {proposal_id}")
        return proposal

    def _ensure_attempt_worker_matches(self, task: Task, worker_id: str | None) -> None:
        if task.worker_id != worker_id:
            raise TaskClaimLost("Work attempt does not carry the task's current worker authority.")
        if worker_id is not None:
            # Task ownership leases use wall-clock time. ``self._clock`` is the
            # independently injectable availability/verifier lifecycle clock.
            _ensure_active_task_lease(task, worker_id)

    def _ensure_attempt_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_attempt_state_is_current(task, attempt)
        self._ensure_attempt_worker_matches(task, attempt.worker_id)

    def _ensure_decision_attempt_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_attempt_state_is_current(task, attempt)
        if task.worker_id not in {None, attempt.worker_id}:
            raise TaskClaimLost(
                "Completion decision conflicts with replacement task-worker authority."
            )

    def _ensure_attempt_state_is_current(self, task: Task, attempt: WorkAttempt) -> None:
        self._ensure_task_contract_matches(task, attempt.contract)
        attempt_ids = self._attempt_ids_by_task.get(task.id, [])
        if not attempt_ids or attempt_ids[-1] != attempt.attempt_id:
            raise WorkCompletionConflict(
                "Work operation does not reference the latest task attempt."
            )
        if task.status is not TaskStatus.RUNNING or task.session_id != attempt.session_id:
            raise WorkCompletionConflict("Work attempt no longer owns the live task session.")

    def _ensure_completion_proposal_is_current(self, proposal: CompletionProposal) -> None:
        attempt = self._require_work_attempt(proposal.attempt_id)
        if attempt.task_id != proposal.task_id or attempt.contract != proposal.contract:
            raise WorkCompletionConflict(
                "Completion proposal conflicts with its durable work attempt."
            )
        task = self._require_task(proposal.task_id)
        self._ensure_attempt_state_is_current(task, attempt)

    def _active_work_contract_task_for_session(self, session_id: str) -> Task | None:
        task_ids = self._contracted_task_ids_by_session.get(session_id)
        if not task_ids:
            return None
        task_id = next(iter(task_ids))
        task = self._tasks.get(task_id)
        if task is None or task.session_id != session_id or task.work_contract is None:
            raise TaskTopologyInconsistent(
                "The in-memory contracted-session authority index is inconsistent."
            )
        return task

    def _ensure_contract_session_accepts_attachment(
        self,
        contract: WorkContractRef | None,
        session_id: str | None,
        *,
        require_session: bool = False,
    ) -> None:
        if contract is None:
            return
        if session_id is None:
            if require_session:
                raise WorkCompletionConflict(
                    "Contracted tasks require a session binding before starting."
                )
            return
        validate_work_completion_linked_id(session_id, "session_id")
        if session_id in self._ordinary_execution_session_ids:
            raise WorkCompletionConflict(
                "Work-contract attachment conflicts with prior ordinary session execution."
            )

    def _task_parent_for_create(
        self,
        request: TaskCreate,
        *,
        task_id: str,
    ) -> TaskInvocationSnapshot | None:
        parent_task_id = request.parent_task_id
        if parent_task_id is None:
            return None
        if parent_task_id == task_id:
            raise ValueError("Task cannot be its own parent.")
        parent = self._tasks.get(parent_task_id)
        if parent is None:
            raise ValueError(f"Parent task not found: {parent_task_id}")
        return TaskInvocationSnapshot(
            id=parent.id,
            session_id=parent.session_id,
            invocation=parent.invocation,
        )

    def _require_owned_leased_task(self, task_id: str, worker_id: str) -> Task:
        task = self._require_task(task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        return task

    def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
        accepted_decision_id: str | None = None,
    ) -> Task:
        updated = self._prepare_finished_task(
            task_id,
            status,
            result=result,
            error=error,
            worker_id=worker_id,
            accepted_decision_id=accepted_decision_id,
            now=datetime.now(UTC),
        )
        self._store_task(updated)
        return updated.model_copy(deep=True)

    def _prepare_finished_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None,
        accepted_decision_id: str | None,
        now: datetime,
    ) -> Task:
        task = self._require_task(task_id)
        if (
            status is TaskStatus.COMPLETED
            and task.work_contract is not None
            and accepted_decision_id is None
        ):
            raise TaskCompletionDecisionRequired(
                "Contracted task completion requires an accepted durable verifier decision."
            )
        if worker_id is not None:
            if task.worker_id != worker_id:
                raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
            _ensure_active_task_lease(task, worker_id, now=now)
        _ensure_can_transition(task, status)
        if task.retry_series is not None:
            if status is not TaskStatus.CANCELLED:
                raise ValueError(
                    "Retry-series tasks require settle_task_retry_attempt for "
                    "completion or failure."
                )
            if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                cancellation_requested = _task_retry_cancellation_requested_task(
                    task,
                    error=error,
                    updated_at=now,
                )
                self._store_task(cancellation_requested)
                return cancellation_requested.model_copy(deep=True)
            cancellation = _cancelled_task_retry_settlement(
                task,
                error=error,
                committed_at=now,
            )
            self._store_task(cancellation.task)
            self._retry_settlements[(cancellation.task_id, cancellation.idempotency_key)] = (
                cancellation
            )
            return cancellation.task.model_copy(deep=True)
        updated = task.model_copy(
            update={
                "status": status,
                "status_reason": None,
                "status_payload": None,
                "result": deepcopy(result),
                "error": deepcopy(error),
                "worker_id": None,
                "lease_expires_at": None,
                "started_at": task.started_at or now,
                "completed_at": now,
                "updated_at": now,
                "retry_series": None,
            }
        )
        return updated

    def _prepare_held_task_from_completion_decision(
        self,
        task: Task,
        status: TaskStatus,
        *,
        decision: CompletionDecision,
        now: datetime,
        status_reason: str | None = None,
    ) -> Task:
        if status not in {TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.NEEDS_ATTENTION}:
            raise ValueError("Completion decisions can only apply supported held statuses.")
        if task.status is not TaskStatus.RUNNING:
            raise WorkCompletionConflict("Completion decision no longer owns a running task.")
        return task.model_copy(
            update={
                "status": status,
                "status_reason": status_reason or f"work_contract_{decision.verdict.value}",
                "status_payload": {
                    "completion_decision_id": decision.decision_id,
                    "gap_fingerprint": decision.gap_fingerprint,
                    "verdict": decision.verdict.value,
                },
                "worker_id": None,
                "lease_expires_at": None,
                "updated_at": now,
            }
        )

    def _completion_rejection_hold(
        self,
        contract: WorkContract,
        decision: CompletionDecision,
        attempt: WorkAttempt,
    ) -> tuple[TaskStatus, str] | None:
        policy = contract.continuation_policy
        if attempt.ordinal >= policy.max_attempts:
            return (TaskStatus.NEEDS_ATTENTION, "work_contract_attempt_limit")

        matching_gap_count = 0
        for attempt_id in self._attempt_ids_by_task.get(decision.task_id, []):
            proposal_id = self._proposal_id_by_attempt.get(attempt_id)
            decision_id = (
                None if proposal_id is None else self._decision_id_by_proposal.get(proposal_id)
            )
            candidate = None if decision_id is None else self._completion_decisions.get(decision_id)
            if (
                candidate is not None
                and candidate.verdict is CompletionVerdict.REJECTED
                and candidate.gap_fingerprint == decision.gap_fingerprint
            ):
                matching_gap_count += 1
        repeated_gap_count = max(0, matching_gap_count - 1)
        if repeated_gap_count >= policy.max_repeated_gap_count:
            return (TaskStatus.NEEDS_ATTENTION, "work_contract_repeated_gap_limit")
        if policy.rejection_action is CompletionRejectionAction.INTERRUPT:
            return (TaskStatus.PAUSED, "work_contract_rejected")
        return None

    async def _hold_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None,
        payload: dict[str, Any] | None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        reason = _copy_optional_status_reason(reason)
        payload = _copy_optional_status_payload(payload)
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_can_hold_task(task, status)
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": status,
                    "status_reason": reason,
                    "status_payload": deepcopy(payload),
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    def _task_topology_candidates(
        self,
        keys: Sequence[tuple[datetime, str]],
        *,
        cursor: str | None,
        scope_kind: Literal["session", "parent_task"],
        scope_id: str,
        limit: int,
    ) -> list[Task]:
        start_index = 0
        if cursor is not None:
            cursor_created_at, cursor_id = decode_task_topology_cursor(
                cursor,
                scope_kind=scope_kind,
                scope_id=scope_id,
            )
            start_index = bisect_right(keys, (cursor_created_at, cursor_id))
        candidates: list[Task] = []
        for _, task_id in keys[start_index : start_index + limit + 1]:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index references a missing task."
                )
            if (scope_kind == "session" and task.session_id != scope_id) or (
                scope_kind == "parent_task" and task.parent_task_id != scope_id
            ):
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index contains a contradictory link."
                )
            candidates.append(task)
        return candidates

    def _store_task(self, task: Task) -> None:
        # ``model_copy(update=...)`` intentionally skips Pydantic validation.
        # Revalidate every contracted lifecycle snapshot at the final in-memory
        # publication boundary so no transition can outgrow the bounded task
        # representation that decision receipts rely on.
        if task.work_contract is not None:
            task = copy_task(task)
        prior = self._tasks.get(task.id)
        if prior is not None and (
            prior.created_at,
            prior.session_id,
            prior.parent_task_id,
            prior.work_contract is None,
        ) == (
            task.created_at,
            task.session_id,
            task.parent_task_id,
            task.work_contract is None,
        ):
            # Lifecycle/status updates are the hot path. They do not change
            # either topology index, so avoid an O(n) list removal/reinsert.
            self._tasks[task.id] = task
            return
        if prior is not None:
            self._remove_task_index_entry(self._task_keys_by_session, prior.session_id, prior)
            self._remove_task_index_entry(self._task_keys_by_parent, prior.parent_task_id, prior)
            self._remove_contracted_session_index_entry(prior)
        self._tasks[task.id] = task
        self._add_task_index_entry(self._task_keys_by_session, task.session_id, task)
        self._add_task_index_entry(self._task_keys_by_parent, task.parent_task_id, task)
        self._add_contracted_session_index_entry(task)

    def _add_contracted_session_index_entry(self, task: Task) -> None:
        if task.session_id is None or task.work_contract is None:
            return
        self._contracted_task_ids_by_session.setdefault(task.session_id, {}).setdefault(
            task.id,
            None,
        )

    def _remove_contracted_session_index_entry(self, task: Task) -> None:
        if task.session_id is None or task.work_contract is None:
            return
        task_ids = self._contracted_task_ids_by_session.get(task.session_id)
        if task_ids is None or task.id not in task_ids:
            raise TaskTopologyInconsistent(
                "The in-memory contracted-session authority index is incomplete."
            )
        del task_ids[task.id]
        if not task_ids:
            del self._contracted_task_ids_by_session[task.session_id]

    @staticmethod
    def _add_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        insort(index.setdefault(scope_id, []), (task.created_at, task.id))

    @staticmethod
    def _remove_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        keys = index.get(scope_id)
        if keys is None:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        key = (task.created_at, task.id)
        position = bisect_left(keys, key)
        if position >= len(keys) or keys[position] != key:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        keys.pop(position)
        if not keys:
            del index[scope_id]


def encode_task_topology_cursor(
    scope_kind: Literal["session", "parent_task"],
    scope_id: str,
    node: TaskTopologyNode,
) -> str:
    """Encode a scope-bound direct-link cursor."""

    if scope_kind not in {"session", "parent_task"}:
        raise ValueError("Invalid task topology cursor scope.")
    scope_id = _bounded_task_topology_text(
        scope_id,
        "scope_id",
        max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    if type(node) is not TaskTopologyNode:
        raise TypeError("Task topology cursors require TaskTopologyNode values.")
    raw = json.dumps(
        [
            scope_kind,
            scope_id,
            node.created_at.astimezone(UTC).isoformat(),
            node.id,
        ],
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    if len(encoded) > TASK_TOPOLOGY_MAX_CURSOR_BYTES:
        raise ValueError("Task topology cursor exceeds its byte limit.")
    return encoded


def decode_task_topology_cursor(
    cursor: str,
    *,
    scope_kind: Literal["session", "parent_task"],
    scope_id: str,
) -> tuple[datetime, str]:
    """Decode a task cursor and reject reuse against a different branch."""

    if scope_kind not in {"session", "parent_task"}:
        raise ValueError("Invalid task topology cursor scope.")
    scope_id = _bounded_task_topology_text(
        scope_id,
        "scope_id",
        max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    try:
        cursor = _bounded_task_topology_text(
            cursor,
            "cursor",
            max_bytes=TASK_TOPOLOGY_MAX_CURSOR_BYTES,
        )
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical task topology cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid task topology cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 4
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
        or type(decoded[2]) is not str
        or type(decoded[3]) is not str
        or decoded[0] != scope_kind
        or decoded[1] != scope_id
        or not decoded[3]
    ):
        raise ValueError("Invalid task topology cursor.")
    try:
        task_id = _bounded_task_topology_text(
            decoded[3],
            "cursor task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )
        created_at = datetime.fromisoformat(decoded[2])
    except ValueError as exc:
        raise ValueError("Invalid task topology cursor.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Invalid task topology cursor.")
    return created_at.astimezone(UTC), task_id


def _validate_task_topology_page(
    tasks: tuple[TaskTopologyNode, ...],
    next_cursor: str | None,
    has_more: bool,
    *,
    scope_id: str,
    scope_kind: Literal["session", "parent_task"],
) -> None:
    task_ids = [task.id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("A task topology branch must not repeat a task.")
    if list(tasks) != sorted(tasks, key=lambda task: (task.created_at, task.id)):
        raise ValueError("Task topology branches must use stable creation ordering.")
    if has_more and next_cursor is None:
        raise ValueError("A task topology branch with more rows requires a cursor.")
    if not has_more and next_cursor is not None:
        raise ValueError("A complete task topology branch cannot expose a cursor.")
    if next_cursor is not None:
        cursor_created_at, cursor_id = decode_task_topology_cursor(
            next_cursor,
            scope_kind=scope_kind,
            scope_id=scope_id,
        )
        if not tasks or (cursor_created_at, cursor_id) != (
            tasks[-1].created_at,
            tasks[-1].id,
        ):
            raise ValueError(
                "A task topology continuation cursor must identify the last returned task."
            )


def build_task_topology_result(
    *,
    observed_at: datetime,
    linked_session_ids: Iterable[str],
    session_branch_candidates: Iterable[Iterable[TaskTopologyNode]],
    session_branch_limits: Iterable[int],
    expanded_parents: Iterable[TaskTopologyNode],
    child_branch_candidates: Iterable[Iterable[TaskTopologyNode]],
    child_branch_limits: Iterable[int],
    session_task_limit: int,
    child_limit: int,
) -> TaskTopologyStoreResult:
    """Apply the shared task-node ceiling without losing branch continuation."""

    for value, field_name in (
        (session_task_limit, "session_task_limit"),
        (child_limit, "child_limit"),
    ):
        if type(value) is not int or value < 1 or value > TASK_TOPOLOGY_MAX_BRANCH_LIMIT:
            raise ValueError(f"{field_name} is outside the task topology bounds.")

    session_ids = tuple(islice(linked_session_ids, TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS + 1))
    if len(session_ids) > TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS:
        raise ValueError("Task topology exceeds the linked-session bound.")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Task topology linked sessions must not contain duplicates.")
    allocated_session_limits = _copy_task_topology_branch_limits(
        session_branch_limits,
        branch_count=len(session_ids),
        requested_limit=session_task_limit,
        field_name="session_branch_limits",
    )
    session_pages = tuple(
        tuple(islice(page, allocated_limit + 1))
        for page, allocated_limit in zip(
            islice(session_branch_candidates, len(session_ids) + 1),
            allocated_session_limits,
            strict=True,
        )
    )
    if len(session_pages) != len(session_ids):
        raise ValueError("Every linked session requires one task candidate page.")

    expanded_nodes = tuple(islice(expanded_parents, TASK_TOPOLOGY_MAX_EXPANDED_PARENTS + 1))
    if len(expanded_nodes) > TASK_TOPOLOGY_MAX_EXPANDED_PARENTS:
        raise ValueError("Task topology exceeds the expanded-parent bound.")
    allocated_child_limits = _copy_task_topology_branch_limits(
        child_branch_limits,
        branch_count=len(expanded_nodes),
        requested_limit=child_limit,
        field_name="child_branch_limits",
    )
    child_pages = tuple(
        tuple(islice(page, allocated_limit + 1))
        for page, allocated_limit in zip(
            islice(child_branch_candidates, len(expanded_nodes) + 1),
            allocated_child_limits,
            strict=True,
        )
    )
    if len(child_pages) != len(expanded_nodes):
        raise ValueError("Every expanded task parent requires one candidate page.")

    all_pages = (*session_pages, *child_pages)
    nonempty_after = [
        sum(bool(later) for later in all_pages[index + 1 :]) for index in range(len(all_pages))
    ]
    retained_ids = {node.id for node in expanded_nodes}

    session_branches: list[TaskTopologySessionBranch] = []
    page_index = 0
    for session_id, candidates, allocated_limit in zip(
        session_ids,
        session_pages,
        allocated_session_limits,
        strict=True,
    ):
        retained = _retain_task_topology_page(
            candidates,
            retained_ids=retained_ids,
            reserve_unique_slots=nonempty_after[page_index],
            limit=allocated_limit,
        )
        page_index += 1
        has_more = len(candidates) > len(retained)
        session_branches.append(
            TaskTopologySessionBranch(
                session_id=session_id,
                tasks=retained,
                next_cursor=(
                    encode_task_topology_cursor("session", session_id, retained[-1])
                    if has_more
                    else None
                ),
                has_more=has_more,
            )
        )

    child_branches: list[TaskTopologyChildBranch] = []
    for parent, candidates, allocated_limit in zip(
        expanded_nodes,
        child_pages,
        allocated_child_limits,
        strict=True,
    ):
        retained = _retain_task_topology_page(
            candidates,
            retained_ids=retained_ids,
            reserve_unique_slots=nonempty_after[page_index],
            limit=allocated_limit,
        )
        page_index += 1
        has_more = len(candidates) > len(retained)
        child_branches.append(
            TaskTopologyChildBranch(
                parent_task_id=parent.id,
                children=retained,
                next_cursor=(
                    encode_task_topology_cursor("parent_task", parent.id, retained[-1])
                    if has_more
                    else None
                ),
                has_more=has_more,
            )
        )

    loaded_nodes = (
        *expanded_nodes,
        *(task for branch in session_branches for task in branch.tasks),
        *(task for branch in child_branches for task in branch.children),
    )
    loaded_nodes_by_id: dict[str, TaskTopologyNode] = {}
    for node in loaded_nodes:
        prior = loaded_nodes_by_id.setdefault(node.id, node)
        if prior != node:
            raise TaskTopologyInconsistent(
                "Task topology contains contradictory representations of one task."
            )
    _reject_loaded_task_topology_cycles(loaded_nodes_by_id)

    return TaskTopologyStoreResult(
        observed_at=observed_at,
        session_branches=tuple(session_branches),
        expanded_parents=expanded_nodes,
        child_branches=tuple(child_branches),
    )


def _copy_task_topology_branch_limits(
    values: Iterable[int],
    *,
    branch_count: int,
    requested_limit: int,
    field_name: str,
) -> tuple[int, ...]:
    copied = tuple(islice(values, branch_count + 1))
    if len(copied) != branch_count:
        raise ValueError(f"{field_name} must provide exactly one limit per branch.")
    if any(type(value) is not int or value < 1 or value > requested_limit for value in copied):
        raise ValueError(f"{field_name} contains an invalid allocated branch limit.")
    return copied


def _retain_task_topology_page(
    candidates: tuple[TaskTopologyNode, ...],
    *,
    retained_ids: set[str],
    reserve_unique_slots: int,
    limit: int,
) -> tuple[TaskTopologyNode, ...]:
    available_unique = TASK_TOPOLOGY_MAX_NODES - len(retained_ids)
    unique_capacity = max(0, available_unique - reserve_unique_slots)
    retained: list[TaskTopologyNode] = []
    new_ids: set[str] = set()
    for candidate in candidates[:limit]:
        is_new = candidate.id not in retained_ids and candidate.id not in new_ids
        if is_new and len(new_ids) >= unique_capacity:
            break
        retained.append(candidate)
        if is_new:
            new_ids.add(candidate.id)
    if candidates and not retained:
        raise RuntimeError("Task topology node allocation could not retain a branch cursor.")
    retained_ids.update(new_ids)
    return tuple(retained)


def _reject_loaded_task_topology_cycles(
    nodes_by_id: Mapping[str, TaskTopologyNode],
) -> None:
    _reject_task_parent_link_cycles(
        {node_id: node.parent_task_id for node_id, node in nodes_by_id.items()}
    )


def _bounded_optional_task_topology_parent_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return _bounded_task_topology_text(
            value,
            "parent_task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task topology record contains an invalid durable parent identifier."
        ) from exc


async def _validate_task_topology_ancestry(
    seed_nodes: Iterable[TaskTopologyNode],
    load_parent_links: Callable[
        [tuple[str, ...]],
        Awaitable[Mapping[str, str | None]],
    ],
) -> None:
    """Validate complete parent chains for projected candidates under hard bounds."""

    parent_by_id: dict[str, str | None] = {}
    for node in seed_nodes:
        prior = parent_by_id.setdefault(node.id, node.parent_task_id)
        if prior != node.parent_task_id:
            raise TaskTopologyInconsistent(
                "Task topology contains contradictory parent links for one task."
            )

    frontier = {
        parent_id
        for parent_id in parent_by_id.values()
        if parent_id is not None and parent_id not in parent_by_id
    }
    depth = 0
    while frontier:
        if depth >= TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH:
            raise TaskTopologyTraversalLimitExceeded(
                "Task topology ancestry exceeds its depth limit."
            )
        task_ids = tuple(sorted(frontier))
        if len(parent_by_id) + len(task_ids) > TASK_TOPOLOGY_MAX_VALIDATION_NODES:
            raise TaskTopologyTraversalLimitExceeded(
                "Task topology ancestry exceeds its validation-node limit."
            )
        loaded = await load_parent_links(task_ids)
        if set(loaded) != set(task_ids):
            raise TaskTopologyInconsistent(
                "A task topology record references a missing durable parent."
            )
        for task_id in task_ids:
            parent_by_id[task_id] = _bounded_optional_task_topology_parent_id(loaded[task_id])
        frontier = {
            parent_id
            for parent_id in (parent_by_id[task_id] for task_id in task_ids)
            if parent_id is not None and parent_id not in parent_by_id
        }
        depth += 1

    _reject_task_parent_link_cycles(parent_by_id)


def _reject_task_parent_link_cycles(
    parent_by_id: Mapping[str, str | None],
) -> None:
    complete: set[str] = set()
    for start_id in parent_by_id:
        if start_id in complete:
            continue
        path: list[str] = []
        path_positions: dict[str, int] = {}
        current_id: str | None = start_id
        while current_id is not None and current_id in parent_by_id:
            if current_id in complete:
                break
            if current_id in path_positions:
                raise TaskTopologyCycle("Task topology contains a cycle among loaded task nodes.")
            path_positions[current_id] = len(path)
            path.append(current_id)
            current_id = parent_by_id[current_id]
        complete.update(path)


def copy_task(task: Task) -> Task:
    if type(task) is not Task:
        raise TypeError("Tasks must be Task instances.")
    if task.work_contract is not None:
        _preflight_bounded_task_payloads(task)
    return Task(
        id=task.id,
        type=task.type,
        title=task.title,
        description=task.description,
        status=task.status,
        session_id=task.session_id,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        status_reason=task.status_reason,
        status_payload=(
            None
            if task.status_payload is None
            else copy_durable_json_object(task.status_payload, "status_payload")
        ),
        input=copy_durable_json_object(task.input, "input"),
        result=(None if task.result is None else copy_durable_json_object(task.result, "result")),
        error=None if task.error is None else copy_durable_json_object(task.error, "error"),
        metadata=copy_durable_json_object(task.metadata, "metadata"),
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        invocation=copy_task_invocation(task.invocation),
        retry_series=(
            None
            if task.retry_series is None
            else _copy_task_retry_series_snapshot(task.retry_series)
        ),
        work_contract=copy_work_contract_ref(task.work_contract),
    )


def _copy_task_retry_policy(policy: TaskRetryPolicy) -> TaskRetryPolicy:
    if type(policy) is not TaskRetryPolicy:
        raise TypeError("Task retry policy must be a TaskRetryPolicy instance.")
    return TaskRetryPolicy(
        max_attempts=policy.max_attempts,
        max_elapsed_seconds=policy.max_elapsed_seconds,
        max_total_tokens=policy.max_total_tokens,
        max_estimated_cost=policy.max_estimated_cost,
        cost_currency=policy.cost_currency,
        initial_backoff_seconds=policy.initial_backoff_seconds,
        backoff_multiplier=policy.backoff_multiplier,
        max_backoff_seconds=policy.max_backoff_seconds,
    )


def _copy_task_retry_series_snapshot(
    series: TaskRetrySeriesSnapshot,
) -> TaskRetrySeriesSnapshot:
    if type(series) is not TaskRetrySeriesSnapshot:
        raise TypeError("Task retry authority must be a TaskRetrySeriesSnapshot instance.")
    return TaskRetrySeriesSnapshot(
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        authority_sha256=series.authority_sha256,
        attempt=series.attempt,
        policy=_copy_task_retry_policy(series.policy),
        started_at=series.started_at,
        cumulative_tokens=series.cumulative_tokens,
        cumulative_estimated_cost=series.cumulative_estimated_cost,
        attempts_remaining=series.attempts_remaining,
        tokens_remaining=series.tokens_remaining,
        estimated_cost_remaining=series.estimated_cost_remaining,
        elapsed_deadline=series.elapsed_deadline,
        disposition=series.disposition,
        predecessor_task_id=series.predecessor_task_id,
        successor_task_id=series.successor_task_id,
        next_eligible_at=series.next_eligible_at,
    )


def _copy_task_retry_event(event: TaskRetryEvent) -> TaskRetryEvent:
    if type(event) is not TaskRetryEvent:
        raise TypeError("Task retry events must be TaskRetryEvent instances.")
    return TaskRetryEvent(
        id=event.id,
        type=event.type,
        task_id=event.task_id,
        series_id=event.series_id,
        causal_budget_id=event.causal_budget_id,
        attempt=event.attempt,
        disposition=event.disposition,
        occurred_at=event.occurred_at,
        attempts_remaining=event.attempts_remaining,
        tokens_remaining=event.tokens_remaining,
        estimated_cost_remaining=event.estimated_cost_remaining,
        cost_currency=event.cost_currency,
        elapsed_deadline=event.elapsed_deadline,
        next_eligible_at=event.next_eligible_at,
    )


def _copy_task_retry_settlement_result(
    receipt: TaskRetrySettlementResult,
) -> TaskRetrySettlementResult:
    if type(receipt) is not TaskRetrySettlementResult:
        raise TypeError(
            "Task retry settlement loads must return TaskRetrySettlementResult instances."
        )
    return TaskRetrySettlementResult(
        task_id=receipt.task_id,
        idempotency_key=receipt.idempotency_key,
        request_sha256=receipt.request_sha256,
        task=copy_task(receipt.task),
        successor=None if receipt.successor is None else copy_task(receipt.successor),
        events=tuple(_copy_task_retry_event(event) for event in receipt.events),
        committed_at=receipt.committed_at,
    )


def prepare_task_terminalization(
    request: TaskTerminalizationRequest,
) -> tuple[TaskTerminalizationRequest, str]:
    """Detach and deterministically digest one validated logical request."""

    if type(request) is not TaskTerminalizationRequest:
        raise TypeError(
            "Task terminalization requests must be TaskTerminalizationRequest instances."
        )
    copied = TaskTerminalizationRequest.model_validate(request.model_dump(mode="python"))
    material = {
        "schema": "cayu.task-terminalization.v1",
        "task_id": copied.task_id,
        "idempotency_key": copied.idempotency_key,
        "worker_id": copied.worker_id,
        "kind": copied.kind.value,
        "result": copied.result,
        "error": copied.error,
    }
    request_sha256 = sha256(
        canonical_durable_json_bytes(material, "task_terminalization")
    ).hexdigest()
    return copied, request_sha256


def prepare_task_retry_settlement(
    request: TaskRetrySettlementRequest,
) -> tuple[TaskRetrySettlementRequest, str]:
    """Detach and digest one typed retry-attempt report."""

    if type(request) is not TaskRetrySettlementRequest:
        raise TypeError("Task retry settlements require a TaskRetrySettlementRequest.")
    copied = TaskRetrySettlementRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-retry-settlement.v1",
                **copied.model_dump(mode="json", warnings=False),
            },
            "task_retry_settlement",
        )
    ).hexdigest()
    return copied, request_sha256


def _replay_task_retry_settlement(
    *,
    request_sha256: str,
    receipt: TaskRetrySettlementResult,
    current_task: Task | None,
) -> TaskRetrySettlementResult:
    receipt = _copy_task_retry_settlement_result(receipt)
    current_task = None if current_task is None else copy_task(current_task)
    if receipt.request_sha256 != request_sha256:
        raise TaskTerminalizationConflict(
            "Task retry settlement idempotency key is bound to another intent."
        )
    if current_task != receipt.task:
        raise TaskTerminalizationConflict(
            "Task retry settlement receipt conflicts with the current terminal attempt."
        )
    return receipt


def _validate_task_retry_settlement_receipt_identity(
    receipt: TaskRetrySettlementResult,
    *,
    request: TaskRetrySettlementRequest,
    request_sha256: str,
) -> TaskRetrySettlementResult:
    receipt = _copy_task_retry_settlement_result(receipt)
    if (
        receipt.task_id != request.task_id
        or receipt.idempotency_key != request.idempotency_key
        or receipt.request_sha256 != request_sha256
    ):
        raise TaskTerminalizationConflict(
            "Task retry settlement receipt conflicts with the requested operation."
        )
    return receipt


def _task_retry_series_id(task_id: str) -> str:
    material = canonical_durable_json_bytes(
        {"schema": "cayu.task-retry-series.v1", "first_task_id": task_id},
        "task_retry_series_id",
    )
    return f"task-retry-series:v1:{sha256(material).hexdigest()}"


def _task_retry_successor_id(series_id: str, attempt: int) -> str:
    material = canonical_durable_json_bytes(
        {"schema": "cayu.task-retry-attempt.v1", "series_id": series_id, "attempt": attempt},
        "task_retry_successor_id",
    )
    return f"task-retry-attempt:v1:{sha256(material).hexdigest()}"


def _task_retry_attempt_authority_sha256(
    *,
    task_id: str,
    task_type: str,
    title: str | None,
    description: str | None,
    parent_task_id: str | None,
    assigned_agent_name: str | None,
    available_at: datetime | None,
    created_at: datetime,
    task_input: dict[str, Any],
    metadata: dict[str, Any],
    invocation: TaskInvocation,
    series_id: str,
    causal_budget_id: str,
    attempt: int,
    policy: TaskRetryPolicy,
    started_at: datetime,
    cumulative_tokens: int,
    cumulative_estimated_cost: Decimal,
    predecessor_task_id: str | None,
) -> str:
    """Bind one attempt to its complete immutable and cumulative authority."""

    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-attempt-authority.v1",
            "task_id": task_id,
            "task_type": task_type,
            "title": title,
            "description": description,
            "parent_task_id": parent_task_id,
            "assigned_agent_name": assigned_agent_name,
            "available_at": (
                None
                if available_at is None
                else normalize_utc_datetime(available_at, "available_at").isoformat()
            ),
            "created_at": normalize_utc_datetime(created_at, "created_at").isoformat(),
            "input": task_input,
            "metadata": metadata,
            "invocation": invocation.model_dump(mode="json", warnings=False),
            "series_id": series_id,
            "causal_budget_id": causal_budget_id,
            "attempt": attempt,
            "policy": policy.model_dump(mode="json", warnings=False),
            "started_at": normalize_utc_datetime(started_at, "started_at").isoformat(),
            "cumulative_tokens": cumulative_tokens,
            "cumulative_estimated_cost": str(cumulative_estimated_cost),
            "predecessor_task_id": predecessor_task_id,
        },
        "task_retry_attempt_authority",
    )
    return sha256(material).hexdigest()


def _task_retry_runtime_idempotency_key(task: Task, operation: str) -> str:
    series = task.retry_series
    if series is None:
        raise ValueError("Task does not belong to a retry series.")
    operation = require_clean_nonblank(operation, "operation")
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-runtime-settlement.v1",
            "operation": operation,
            "series_id": series.series_id,
            "task_id": task.id,
            "attempt": series.attempt,
        },
        "task_retry_runtime_idempotency_key",
    )
    return f"task-retry-{operation}:v1:{sha256(material).hexdigest()}"


def _task_retry_cancellation_requested(task: Task) -> bool:
    """Return whether an owned retry attempt carries a durable cancel request."""

    return task.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON


def _task_retry_cancellation_requested_task(
    task: Task,
    *,
    error: dict[str, Any] | None,
    updated_at: datetime,
) -> Task:
    """Fence an active attempt while its worker proves dispatched work quiescent."""

    series = task.retry_series
    if (
        series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id is None
        or task.lease_expires_at is None
    ):
        raise TaskTerminalizationConflict("Task retry attempt cannot drain cancellation.")
    if _task_retry_cancellation_requested(task):
        return task.model_copy(deep=True)
    cancellation_error = (
        {"code": TaskRetrySeriesDisposition.CANCELLED.value}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    return task.model_copy(
        update={
            "status_reason": _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
            "status_payload": {
                "settlement_idempotency_key": _task_retry_runtime_idempotency_key(
                    task,
                    "cancellation",
                ),
                "error": cancellation_error,
            },
            "updated_at": normalize_utc_datetime(updated_at, "updated_at"),
        },
        deep=True,
    )


def _task_retry_requested_cancellation_settlement(
    task: Task,
    *,
    worker_id: str,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementRequest | None:
    """Reconstruct the exact cancellation intent owned by a draining attempt."""

    if not _task_retry_cancellation_requested(task):
        return None
    series = task.retry_series
    payload = task.status_payload
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.worker_id != worker_id
        or type(payload) is not dict
        or set(payload) != {"settlement_idempotency_key", "error"}
        or type(payload.get("settlement_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
    ):
        raise TaskTerminalizationConflict(
            "Task retry cancellation request conflicts with active ownership."
        )
    expected_key = _task_retry_runtime_idempotency_key(task, "cancellation")
    if payload["settlement_idempotency_key"] != expected_key:
        raise TaskTerminalizationConflict(
            "Task retry cancellation request conflicts with its attempt identity."
        )
    return TaskRetrySettlementRequest(
        task_id=task.id,
        worker_id=worker_id,
        idempotency_key=expected_key,
        causal_budget_id=series.causal_budget_id,
        disposition=TaskRetryAttemptDisposition.CANCELLED,
        error=copy_durable_json_object(payload["error"], "error"),
        token_count=token_count,
        estimated_cost=estimated_cost,
    )


def _validated_task_retry_terminal_accounting(
    *,
    token_count: int,
    estimated_cost: Decimal,
) -> tuple[int, Decimal]:
    """Validate the accounting copied into a runtime-owned terminal settlement."""

    report = TaskRetryAttemptReport(
        idempotency_key="task-retry-terminal-accounting",
        disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        error={"code": "runtime_terminal"},
        token_count=token_count,
        estimated_cost=estimated_cost,
    )
    return report.token_count, report.estimated_cost


def _task_retry_attempt_elapsed(task: Task, *, series_now: datetime) -> bool:
    series_now = normalize_utc_datetime(series_now, "series_now")
    series = task.retry_series
    return bool(
        task.status is TaskStatus.PENDING
        and task.session_id is None
        and series is not None
        and series.disposition is TaskRetrySeriesDisposition.ACTIVE
        and series.elapsed_deadline is not None
        and series.elapsed_deadline <= series_now
    )


def _claimed_task_retry_attempt_elapsed(task: Task, *, series_now: datetime) -> bool:
    series_now = normalize_utc_datetime(series_now, "series_now")
    series = task.retry_series
    return bool(
        task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        and task.session_id is None
        and not _task_retry_cancellation_requested(task)
        and series is not None
        and series.disposition is TaskRetrySeriesDisposition.ACTIVE
        and series.elapsed_deadline is not None
        and series.elapsed_deadline <= series_now
    )


def _elapsed_claimed_task_retry_settlement(
    task: Task,
    *,
    committed_at: datetime,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult:
    """Finalize a live claimed attempt after store time proves elapsed authority."""

    return _runtime_task_retry_terminal_settlement(
        task,
        operation="elapsed",
        request_disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        series_disposition=TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        status=TaskStatus.FAILED,
        error={"code": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED.value},
        committed_at=committed_at,
        token_count=token_count,
        estimated_cost=estimated_cost,
    )


def _expired_task_retry_settlement(
    task: Task,
    *,
    committed_at: datetime,
    series_now: datetime,
) -> TaskRetrySettlementResult:
    """Finalize an unclaimed attempt whose cumulative elapsed authority expired."""

    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    series_now = normalize_utc_datetime(series_now, "series_now")
    if not _task_retry_attempt_elapsed(task, series_now=series_now):
        raise TaskTerminalizationConflict("Task retry attempt has not elapsed.")
    return _runtime_task_retry_terminal_settlement(
        task,
        operation="expiration",
        request_disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        series_disposition=TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        status=TaskStatus.FAILED,
        error={"code": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED.value},
        committed_at=committed_at,
    )


def _cancelled_task_retry_settlement(
    task: Task,
    *,
    error: dict[str, Any] | None,
    committed_at: datetime,
) -> TaskRetrySettlementResult:
    series = task.retry_series
    if (
        series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.status in _TERMINAL_TASK_STATUSES
    ):
        raise TaskTerminalizationConflict("Task retry attempt is not cancellable.")
    return _runtime_task_retry_terminal_settlement(
        task,
        operation="cancellation",
        request_disposition=TaskRetryAttemptDisposition.CANCELLED,
        series_disposition=TaskRetrySeriesDisposition.CANCELLED,
        status=TaskStatus.CANCELLED,
        error=(
            {"code": TaskRetrySeriesDisposition.CANCELLED.value}
            if error is None
            else copy_durable_json_object(error, "error")
        ),
        committed_at=committed_at,
    )


def _runtime_task_retry_terminal_settlement(
    task: Task,
    *,
    operation: str,
    request_disposition: TaskRetryAttemptDisposition,
    series_disposition: TaskRetrySeriesDisposition,
    status: TaskStatus,
    error: dict[str, Any],
    committed_at: datetime,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult:
    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    series = task.retry_series
    if series is None or series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    request, request_sha256 = _task_retry_runtime_terminal_request(
        task,
        operation=operation,
        request_disposition=request_disposition,
        error=error,
        token_count=token_count,
        estimated_cost=estimated_cost,
    )
    idempotency_key = request.idempotency_key
    cumulative_tokens = series.cumulative_tokens + request.token_count
    if cumulative_tokens > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Task retry cumulative token count exceeds durable bounds.")
    cumulative_estimated_cost = series.cumulative_estimated_cost + request.estimated_cost
    _bounded_task_retry_decimal(
        cumulative_estimated_cost,
        "cumulative_estimated_cost",
        max_digits=_TASK_RETRY_TOTAL_COST_MAX_DIGITS,
    )
    settled_authority_sha256 = _task_retry_attempt_authority_sha256(
        task_id=task.id,
        task_type=task.type,
        title=task.title,
        description=task.description,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        created_at=task.created_at,
        task_input=task.input,
        metadata=task.metadata,
        invocation=task.invocation,
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled_series = _task_retry_series_snapshot(
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        authority_sha256=settled_authority_sha256,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        disposition=series_disposition,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled = task.model_copy(
        update={
            "status": status,
            "status_reason": series_disposition.value,
            "status_payload": {
                "retry_series_id": series.series_id,
                "attempt": series.attempt,
                "disposition": series_disposition.value,
                "settlement_idempotency_key": idempotency_key,
                "causal_budget_id": series.causal_budget_id,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_estimated_cost": str(cumulative_estimated_cost),
                "cost_currency": series.policy.cost_currency,
                "next_eligible_at": None,
            },
            "result": None,
            "error": deepcopy(error),
            "worker_id": None,
            "lease_expires_at": None,
            "started_at": task.started_at or committed_at,
            "completed_at": committed_at,
            "updated_at": committed_at,
            "retry_series": settled_series,
        },
        deep=True,
    )
    return TaskRetrySettlementResult(
        task_id=task.id,
        idempotency_key=idempotency_key,
        request_sha256=request_sha256,
        task=settled,
        events=_task_retry_events(settled, occurred_at=committed_at),
        committed_at=committed_at,
    )


def _task_retry_runtime_terminal_request(
    task: Task,
    *,
    operation: str,
    request_disposition: TaskRetryAttemptDisposition,
    error: dict[str, Any],
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> tuple[TaskRetrySettlementRequest, str]:
    """Build the exact request identity shared by mutation and reconciliation."""

    series = task.retry_series
    if series is None or series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    operation = require_clean_nonblank(operation, "operation")
    return prepare_task_retry_settlement(
        TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=f"cayu-runtime-retry-{operation}",
            idempotency_key=_task_retry_runtime_idempotency_key(task, operation),
            causal_budget_id=series.causal_budget_id,
            disposition=request_disposition,
            error=error,
            token_count=token_count,
            estimated_cost=estimated_cost,
            retry_after_seconds=(
                0.0
                if request_disposition is TaskRetryAttemptDisposition.RETRYABLE_FAILURE
                else None
            ),
        )
    )


def _task_retry_backoff_seconds(policy: TaskRetryPolicy, completed_attempt: int) -> float:
    delay = policy.initial_backoff_seconds
    for _ in range(max(0, completed_attempt - 1)):
        if delay >= policy.max_backoff_seconds:
            return policy.max_backoff_seconds
        delay = min(policy.max_backoff_seconds, delay * policy.backoff_multiplier)
    return min(delay, policy.max_backoff_seconds)


def _task_retry_series_snapshot(
    *,
    series_id: str,
    causal_budget_id: str,
    authority_sha256: str,
    attempt: int,
    policy: TaskRetryPolicy,
    started_at: datetime,
    cumulative_tokens: int = 0,
    cumulative_estimated_cost: Decimal = Decimal(0),
    disposition: TaskRetrySeriesDisposition = TaskRetrySeriesDisposition.ACTIVE,
    predecessor_task_id: str | None = None,
    successor_task_id: str | None = None,
    next_eligible_at: datetime | None = None,
) -> TaskRetrySeriesSnapshot:
    return TaskRetrySeriesSnapshot(
        series_id=series_id,
        causal_budget_id=causal_budget_id,
        authority_sha256=authority_sha256,
        attempt=attempt,
        policy=policy,
        started_at=started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        attempts_remaining=max(0, policy.max_attempts - attempt),
        tokens_remaining=(
            None
            if policy.max_total_tokens is None
            else max(0, policy.max_total_tokens - cumulative_tokens)
        ),
        estimated_cost_remaining=(
            None
            if policy.max_estimated_cost is None
            else max(Decimal(0), policy.max_estimated_cost - cumulative_estimated_cost)
        ),
        elapsed_deadline=(
            None
            if policy.max_elapsed_seconds is None
            else started_at + timedelta(seconds=policy.max_elapsed_seconds)
        ),
        disposition=disposition,
        predecessor_task_id=predecessor_task_id,
        successor_task_id=successor_task_id,
        next_eligible_at=next_eligible_at,
    )


def _task_retry_event_id(
    *,
    series_id: str,
    attempt: int,
    event_type: TaskRetryEventType,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-event.v1",
            "series_id": series_id,
            "attempt": attempt,
            "type": event_type.value,
        },
        "task_retry_event_id",
    )
    return f"task-retry-event:v1:{sha256(material).hexdigest()}"


def _task_retry_events(
    task: Task,
    *,
    occurred_at: datetime,
) -> tuple[TaskRetryEvent, TaskRetryEvent]:
    series = task.retry_series
    if series is None:  # pragma: no cover - internal construction invariant
        raise AssertionError("Retry settlement event requires retry-series evidence.")
    event_fields = {
        "task_id": task.id,
        "series_id": series.series_id,
        "causal_budget_id": series.causal_budget_id,
        "attempt": series.attempt,
        "disposition": series.disposition,
        "occurred_at": occurred_at,
        "attempts_remaining": series.attempts_remaining,
        "tokens_remaining": series.tokens_remaining,
        "estimated_cost_remaining": series.estimated_cost_remaining,
        "cost_currency": series.policy.cost_currency,
        "elapsed_deadline": series.elapsed_deadline,
        "next_eligible_at": series.next_eligible_at,
    }
    outcome_type = (
        TaskRetryEventType.RETRY_SCHEDULED
        if series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        else TaskRetryEventType.SERIES_TERMINAL
    )
    return (
        TaskRetryEvent(
            id=_task_retry_event_id(
                series_id=series.series_id,
                attempt=series.attempt,
                event_type=TaskRetryEventType.ATTEMPT_SETTLED,
            ),
            type=TaskRetryEventType.ATTEMPT_SETTLED,
            **event_fields,
        ),
        TaskRetryEvent(
            id=_task_retry_event_id(
                series_id=series.series_id,
                attempt=series.attempt,
                event_type=outcome_type,
            ),
            type=outcome_type,
            **event_fields,
        ),
    )


def _settled_task_retry_attempt(
    task: Task,
    request: TaskRetrySettlementRequest,
    *,
    now: datetime,
    series_now: datetime,
) -> tuple[Task, Task | None]:
    series = task.retry_series
    if series is None:
        raise ValueError("Task does not belong to a retry series.")
    if series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    now = normalize_utc_datetime(now, "now")
    series_now = normalize_utc_datetime(series_now, "series_now")
    if request.causal_budget_id != series.causal_budget_id:
        raise TaskTerminalizationConflict(
            "Task retry settlement conflicts with the series causal budget."
        )
    requested_cancellation = _task_retry_requested_cancellation_settlement(
        task,
        worker_id=request.worker_id,
        token_count=request.token_count,
        estimated_cost=request.estimated_cost,
    )
    if requested_cancellation is not None and request != requested_cancellation:
        raise TaskTerminalizationConflict("Task retry attempt has a pending cancellation request.")
    token_authority_exceeded = (
        series.tokens_remaining is not None and request.token_count > series.tokens_remaining
    )
    cost_authority_exceeded = (
        series.estimated_cost_remaining is not None
        and request.estimated_cost > series.estimated_cost_remaining
    )
    cumulative_tokens = series.cumulative_tokens + request.token_count
    if cumulative_tokens > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Task retry cumulative token count exceeds durable bounds.")
    cumulative_cost = series.cumulative_estimated_cost + request.estimated_cost
    _bounded_task_retry_decimal(
        cumulative_cost,
        "cumulative_estimated_cost",
        max_digits=_TASK_RETRY_TOTAL_COST_MAX_DIGITS,
    )

    disposition: TaskRetrySeriesDisposition
    successor: Task | None = None
    next_eligible_at: datetime | None = None
    successor_task_id: str | None = None
    outcome_result = deepcopy(request.result)
    outcome_error = deepcopy(request.error)
    elapsed_deadline = series.elapsed_deadline
    if requested_cancellation is not None:
        status = TaskStatus.CANCELLED
        disposition = TaskRetrySeriesDisposition.CANCELLED
    elif elapsed_deadline is not None and series_now >= elapsed_deadline:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif token_authority_exceeded:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif cost_authority_exceeded:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.COST_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif request.disposition is TaskRetryAttemptDisposition.SUCCEEDED:
        status = TaskStatus.COMPLETED
        disposition = TaskRetrySeriesDisposition.SUCCEEDED
    elif request.disposition is TaskRetryAttemptDisposition.CANCELLED:
        status = TaskStatus.CANCELLED
        disposition = TaskRetrySeriesDisposition.CANCELLED
    elif request.disposition is TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
    else:
        status = TaskStatus.FAILED
        policy = series.policy
        backoff_seconds = max(
            _task_retry_backoff_seconds(policy, series.attempt),
            request.retry_after_seconds or 0.0,
        )
        next_eligible_at = series_now + timedelta(seconds=backoff_seconds)
        if series.attempt >= policy.max_attempts:
            disposition = TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
        elif elapsed_deadline is not None and next_eligible_at >= elapsed_deadline:
            disposition = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        elif policy.max_total_tokens is not None and cumulative_tokens >= policy.max_total_tokens:
            disposition = TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
        elif policy.max_estimated_cost is not None and cumulative_cost >= policy.max_estimated_cost:
            disposition = TaskRetrySeriesDisposition.COST_EXHAUSTED
        else:
            disposition = TaskRetrySeriesDisposition.RETRY_SCHEDULED
            successor_task_id = _task_retry_successor_id(series.series_id, series.attempt + 1)

    settled_authority_sha256 = _task_retry_attempt_authority_sha256(
        task_id=task.id,
        task_type=task.type,
        title=task.title,
        description=task.description,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        created_at=task.created_at,
        task_input=task.input,
        metadata=task.metadata,
        invocation=task.invocation,
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_cost,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled_series = _task_retry_series_snapshot(
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        authority_sha256=settled_authority_sha256,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_cost,
        disposition=disposition,
        predecessor_task_id=series.predecessor_task_id,
        successor_task_id=successor_task_id,
        next_eligible_at=next_eligible_at if successor_task_id is not None else None,
    )
    settled = task.model_copy(
        update={
            "status": status,
            "status_reason": disposition.value,
            "status_payload": {
                "retry_series_id": series.series_id,
                "attempt": series.attempt,
                "disposition": disposition.value,
                "settlement_idempotency_key": request.idempotency_key,
                "causal_budget_id": series.causal_budget_id,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_estimated_cost": str(cumulative_cost),
                "cost_currency": series.policy.cost_currency,
                "next_eligible_at": (
                    None
                    if settled_series.next_eligible_at is None
                    else settled_series.next_eligible_at.isoformat()
                ),
            },
            "result": outcome_result,
            "error": outcome_error,
            "worker_id": None,
            "lease_expires_at": None,
            "started_at": task.started_at or now,
            "completed_at": now,
            "updated_at": now,
            "retry_series": settled_series,
        },
        deep=True,
    )
    if successor_task_id is not None:
        successor_authority_sha256 = _task_retry_attempt_authority_sha256(
            task_id=successor_task_id,
            task_type=task.type,
            title=task.title,
            description=task.description,
            parent_task_id=task.parent_task_id,
            assigned_agent_name=task.assigned_agent_name,
            available_at=next_eligible_at,
            created_at=now,
            task_input=task.input,
            metadata=task.metadata,
            invocation=task.invocation,
            series_id=series.series_id,
            causal_budget_id=series.causal_budget_id,
            attempt=series.attempt + 1,
            policy=series.policy,
            started_at=series.started_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_estimated_cost=cumulative_cost,
            predecessor_task_id=task.id,
        )
        successor_series = _task_retry_series_snapshot(
            series_id=series.series_id,
            causal_budget_id=series.causal_budget_id,
            authority_sha256=successor_authority_sha256,
            attempt=series.attempt + 1,
            policy=series.policy,
            started_at=series.started_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_estimated_cost=cumulative_cost,
            predecessor_task_id=task.id,
        )
        successor = Task(
            id=successor_task_id,
            type=task.type,
            title=task.title,
            description=task.description,
            status=TaskStatus.PENDING,
            parent_task_id=task.parent_task_id,
            assigned_agent_name=task.assigned_agent_name,
            available_at=next_eligible_at,
            input=copy_durable_json_object(task.input, "input"),
            metadata=copy_durable_json_object(task.metadata, "metadata"),
            created_at=now,
            updated_at=now,
            invocation=copy_task_invocation(task.invocation),
            retry_series=successor_series,
        )
    return settled, successor


def prepare_task_terminalization_receipt_lookup(
    task_id: str,
    idempotency_key: str,
) -> tuple[str, str]:
    return (
        require_clean_nonblank(task_id, "task_id"),
        _validate_task_terminalization_idempotency_key(idempotency_key),
    )


def _validate_task_terminalization_idempotency_key(value: str) -> str:
    value = require_clean_nonblank(value, "idempotency_key")
    if len(value.encode("utf-8")) > TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES:
        raise ValueError(
            "idempotency_key must be at most "
            f"{TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _replay_task_terminalization_receipt(
    *,
    request_sha256: str,
    receipt: TaskTerminalizationReceipt,
    current_task: Task | None,
) -> Task:
    """Validate durable replay proof and return a detached terminal task."""

    if receipt.request_sha256 != request_sha256:
        raise TaskTerminalizationConflict(
            "Task terminalization idempotency key conflicts with another intent."
        )
    if current_task != receipt.task:
        raise TaskTerminalizationConflict(
            "Task terminalization receipt conflicts with the current terminal task."
        )
    return receipt.task.model_copy(deep=True)


async def terminalize_task_with_retry(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskTerminalizationRetryResult:
    """Terminalize once, reconciling only acknowledgement-ambiguous failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_idempotent_terminalization:
        raise ValueError("task_store must support idempotent task terminalization and receipts.")
    request, request_sha256 = prepare_task_terminalization(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(policy.model_dump(mode="python"))

    clock = asyncio.get_running_loop()
    started_at = clock.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        attempt_request = TaskTerminalizationRequest.model_validate(
            request.model_dump(mode="python")
        )
        try:
            task = await asyncio.wait_for(
                task_store.terminalize_task(attempt_request),
                timeout=policy.attempt_timeout_seconds,
            )
            return TaskTerminalizationRetryResult(
                task=task,
                attempt_count=attempt,
                receipt_reconciled=False,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_terminalization_receipt(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            if type(receipt) is not TaskTerminalizationReceipt:
                raise TypeError(
                    "Task terminalization receipt loads must return "
                    "TaskTerminalizationReceipt instances."
                )
            if (
                receipt.task_id != request.task_id
                or receipt.idempotency_key != request.idempotency_key
                or receipt.worker_id != request.worker_id
                or receipt.kind is not request.kind
                or receipt.request_sha256 != request_sha256
            ):
                raise TaskTerminalizationConflict(
                    "Task terminalization receipt conflicts with the retry request."
                )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                if current_task is not None and type(current_task) is not Task:
                    raise TypeError("Task loads must return Task instances.")
                reconciled_task = _replay_task_terminalization_receipt(
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )
                return TaskTerminalizationRetryResult(
                    task=reconciled_task,
                    attempt_count=attempt,
                    receipt_reconciled=True,
                    elapsed_seconds=max(0.0, clock.time() - started_at),
                    applied_backoff_seconds=applied_backoff_seconds,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task terminalization retry loop exited without an outcome.")


async def settle_task_retry_attempt_with_retry(
    task_store: TaskStore,
    request: TaskRetrySettlementRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskRetrySettlementResult:
    """Settle once, reconciling only acknowledgement-ambiguous store failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_task_retry_series:
        raise ValueError("task_store must support atomic task retry-series settlement.")
    request, request_sha256 = prepare_task_retry_settlement(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(
            policy.model_dump(mode="python", warnings=False)
        )

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        try:
            receipt = await asyncio.wait_for(
                task_store.settle_task_retry_attempt(
                    TaskRetrySettlementRequest.model_validate(
                        request.model_dump(mode="python", warnings=False)
                    )
                ),
                timeout=policy.attempt_timeout_seconds,
            )
            return _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_retry_settlement(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            receipt = _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                return _replay_task_retry_settlement(
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, loop.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task retry settlement loop exited without an outcome.")


async def _terminalize_claimed_task(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> Task:
    """Use receipt-safe terminalization when supported, with a legacy fallback."""

    if task_store.supports_idempotent_terminalization:
        return (await terminalize_task_with_retry(task_store, request)).task

    request, _request_sha256 = prepare_task_terminalization(request)
    if request.kind is TaskTerminalKind.COMPLETED:
        if request.result is None:  # pragma: no cover - enforced by the request model
            raise AssertionError("Completed task terminalization requires a result.")
        return await task_store.complete_task(
            request.task_id,
            request.result,
            worker_id=request.worker_id,
        )
    if request.error is None:  # pragma: no cover - enforced by the request model
        raise AssertionError("Failed task terminalization requires an error.")
    return await task_store.fail_task(
        request.task_id,
        request.error,
        worker_id=request.worker_id,
    )


async def _terminalize_claimed_task_or_detect_peer_winner(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> bool:
    """Terminalize the claim, or conservatively identify a peer winner.

    ``True`` means the task is already terminal and this request's key has no
    receipt, so another terminalization won. A receipt under this request's key
    remains an explicit conflict because it may prove changed intent.
    """

    request, _request_sha256 = prepare_task_terminalization(request)
    try:
        await _terminalize_claimed_task(task_store, request)
    except TaskTerminalizationConflict:
        if not task_store.supports_idempotent_terminalization:
            raise
        receipt = await task_store.load_task_terminalization_receipt(
            request.task_id,
            request.idempotency_key,
        )
        if receipt is not None:
            raise
        task = await task_store.load_task(request.task_id)
        if task is not None and task.status in _TERMINAL_TASK_STATUSES:
            return True
        raise
    return False


def _task_terminalization_error_is_acknowledgement_ambiguous(exc: Exception) -> bool:
    if isinstance(
        exc,
        (TaskClaimLost, TaskTerminalizationConflict, TypeError, ValueError),
    ):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    for error_type in type(exc).__mro__:
        module = error_type.__module__
        name = error_type.__name__
        if module == "sqlite3" and name == "OperationalError":
            error_name = getattr(exc, "sqlite_errorname", None)
            return isinstance(error_name, str) and (
                error_name == "SQLITE_IOERR" or error_name.startswith("SQLITE_IOERR_")
            )
        if module.startswith("psycopg") and name == "OperationalError":
            sqlstate = getattr(exc, "sqlstate", None)
            return sqlstate is None or (isinstance(sqlstate, str) and sqlstate.startswith("08"))
    return False


def _task_terminalization_error_category(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    return "database_operational"


def _bounded_task_terminalization_evidence(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES:
        return value
    suffix = f"...[sha256:{sha256(encoded).hexdigest()[:8]}]"
    prefix_bytes = TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES - len(suffix.encode("utf-8"))
    prefix = encoded[:prefix_bytes].decode("utf-8", "ignore")
    return f"{prefix}{suffix}"


def copy_task_create(request: TaskCreate) -> TaskCreate:
    if type(request) is not TaskCreate:
        raise TypeError("Task creation requires a TaskCreate instance.")
    if request.work_contract is not None:
        _preflight_bounded_task_payloads(request, ("input", "metadata"))
    copied = TaskCreate(
        task_id=request.task_id,
        type=request.type,
        title=request.title,
        description=request.description,
        session_id=request.session_id,
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        available_at=request.available_at,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        retry_policy=(
            None
            if request.retry_policy is None
            else TaskRetryPolicy.model_validate(
                request.retry_policy.model_dump(mode="python", warnings=False)
            )
        ),
        work_contract=copy_work_contract_ref(request.work_contract),
        invocation_origin=copy_invocation_origin_claim(request.invocation_origin),
    )
    copied._verified_invocation_origin = (
        None
        if request._verified_invocation_origin is None
        else copy_invocation_origin(request._verified_invocation_origin)
    )
    copied._runtime_invocation_source = request._runtime_invocation_source
    copied._runtime_session_binding = _copy_optional_session_binding(
        request._runtime_session_binding
    )
    return copied


def task_create_with_runtime_invocation(
    request: TaskCreate,
    *,
    source: TaskExecutionSource,
    verified_origin: InvocationOrigin | None = None,
    session_invocation: SessionInvocationBinding | None = None,
) -> TaskCreate:
    """Attach provenance authority minted by a trusted Cayu boundary."""

    if type(request) is not TaskCreate:
        raise TypeError("Runtime task invocation authority requires a TaskCreate request.")
    if type(source) is not TaskExecutionSource:
        raise TypeError("source must be a TaskExecutionSource.")
    if verified_origin is not None:
        verified_origin = copy_invocation_origin(verified_origin)
        if verified_origin.trust is not InvocationOriginTrust.SERVER_VERIFIED:
            raise ValueError("Runtime-verified task origins must use server_verified trust.")
        if request.invocation_origin is not None:
            raise ValueError("A verified task cannot also carry a host origin claim.")
    session_binding = _copy_optional_session_binding(session_invocation)
    if session_binding is not None and (
        request.invocation_origin is not None or verified_origin is not None
    ):
        raise ValueError("Session-derived tasks must inherit their root invocation origin.")
    copied = copy_task_create(request)
    copied._runtime_invocation_source = source
    copied._verified_invocation_origin = verified_origin
    copied._runtime_session_binding = session_binding
    return copied


def task_create_with_execution_source(
    request: TaskCreate,
    *,
    source: TaskExecutionSource,
) -> TaskCreate:
    """Classify work at a trusted direct-SDK host boundary.

    The source is private model state rather than request JSON. Server-owned
    sources are intentionally rejected; only Cayu's server/runtime adapters may
    mint those classifications.
    """

    if type(source) is not TaskExecutionSource:
        raise TypeError("source must be a TaskExecutionSource.")
    if source not in {
        TaskExecutionSource.SDK_TASK,
        TaskExecutionSource.SCHEDULED,
        TaskExecutionSource.WEBHOOK,
    }:
        raise ValueError("Direct SDK task sources must be sdk_task, scheduled, or webhook.")
    return task_create_with_runtime_invocation(request, source=source)


def task_invocation_for_create(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding | None = None,
) -> TaskInvocation:
    """Derive exact provenance inside the atomic task-store create boundary."""

    if type(request) is not TaskCreate:
        raise TypeError("Task invocation derivation requires a TaskCreate request.")
    task_id = require_clean_nonblank(task_id, "task_id")
    source = request._runtime_invocation_source or TaskExecutionSource.SDK_TASK
    verified_origin = request._verified_invocation_origin
    request_session_binding = request._runtime_session_binding
    supplied_session_binding = _copy_optional_session_binding(session_invocation)
    if (
        request_session_binding is not None
        and supplied_session_binding is not None
        and request_session_binding != supplied_session_binding
    ):
        raise ValueError("Task creation carries contradictory session invocation bindings.")
    session_binding = supplied_session_binding or request_session_binding
    if request.invocation_origin is not None and verified_origin is not None:
        raise ValueError("A task cannot carry both host-asserted and verified origins.")
    if parent_task is not None:
        if type(parent_task) not in {Task, TaskInvocationSnapshot}:
            raise TypeError("Parent task provenance must be a task or invocation snapshot.")
        if request.parent_task_id != parent_task.id:
            raise ValueError("Parent task identity conflicts with invocation derivation.")
        if request.invocation_origin is not None or verified_origin is not None:
            raise ValueError("Derived tasks must inherit their root invocation origin.")
        if (
            session_binding is not None
            and request.session_id is not None
            and request.session_id != session_binding.id
        ):
            raise ValueError("Task session identity conflicts with its provenance binding.")
        if session_binding is not None and (
            parent_task.invocation.origin != session_binding.invocation.origin
            or parent_task.invocation.root_invocation_id
            != session_binding.invocation.root_invocation_id
        ):
            raise ValueError("Parent task and attached session invocation provenance conflict.")
        return inherited_task_invocation(
            parent_task.invocation,
            source=source,
            root_session_id=(
                None if session_binding is None else session_binding.invocation.root_session_id
            ),
        )
    if request.parent_task_id is not None:
        raise ValueError("Parent task not found for invocation provenance.")
    if session_binding is not None:
        if request.invocation_origin is not None or verified_origin is not None:
            raise ValueError("Session-derived tasks must inherit their root invocation origin.")
        if request.session_id is not None and request.session_id != session_binding.id:
            raise ValueError("Task session identity conflicts with its provenance binding.")
        return inherited_task_invocation(
            session_binding.invocation,
            source=source,
        )
    if source is TaskExecutionSource.TASK_DISPATCH:
        raise ValueError("Task dispatch provenance requires a parent task or session.")
    if verified_origin is not None:
        if source not in {TaskExecutionSource.HTTP_RUN, TaskExecutionSource.PRODUCT_OPERATION}:
            raise ValueError("Verified task origins require a server-owned task source.")
        origin = copy_invocation_origin(verified_origin)
    elif request.invocation_origin is not None:
        if source not in {
            TaskExecutionSource.SDK_TASK,
            TaskExecutionSource.SCHEDULED,
            TaskExecutionSource.WEBHOOK,
        }:
            raise ValueError("Host-asserted task origins require a trusted host source.")
        origin = InvocationOrigin(
            trust=InvocationOriginTrust.HOST_ASSERTED,
            subject=request.invocation_origin.subject,
            tenant=request.invocation_origin.tenant,
        )
    else:
        if source not in {
            TaskExecutionSource.SDK_TASK,
            TaskExecutionSource.SCHEDULED,
            TaskExecutionSource.WEBHOOK,
        }:
            raise ValueError(f"{source.value} task provenance requires a trusted origin.")
        origin = InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
    return TaskInvocation(
        origin=origin,
        root_invocation_id=str(uuid4()),
        root_session_id=request.session_id,
        source=source,
    )


def _copy_optional_session_binding(
    value: SessionInvocationBinding | None,
) -> SessionInvocationBinding | None:
    if value is None:
        return None
    return copy_session_invocation_binding(value)


def _copy_required_session_binding(
    value: SessionInvocationBinding,
) -> SessionInvocationBinding:
    if value is None:
        raise TypeError("Running task creation requires session invocation provenance.")
    return copy_session_invocation_binding(value)


def _task_invocation_for_attachment(
    task_invocation: TaskInvocation,
    *,
    session_id: str | None,
    session_binding: SessionInvocationBinding | None,
) -> TaskInvocation:
    task_invocation = copy_task_invocation(task_invocation)
    if session_id is None:
        if session_binding is not None:
            raise ValueError("Session provenance binding requires a session_id attachment.")
        return task_invocation
    if session_binding is None:
        raise ValueError("Session provenance binding is required to attach this task.")
    if session_binding.id != session_id:
        raise ValueError("Task session identity conflicts with its provenance binding.")
    session_invocation = session_binding.invocation
    if (
        task_invocation.origin != session_invocation.origin
        or task_invocation.root_invocation_id != session_invocation.root_invocation_id
    ):
        raise ValueError("Task and session invocation provenance conflict.")
    if (
        task_invocation.root_session_id is not None
        and task_invocation.root_session_id != session_invocation.root_session_id
    ):
        raise ValueError("Task and session root identities conflict.")
    return task_invocation


def _task_session_id_for_start(
    *,
    task_id: str,
    stored_session_id: str | None,
    requested_session_id: str | None,
) -> str | None:
    """Resolve one start transition's canonical session without allowing reassignment."""

    task_id = require_clean_nonblank(task_id, "task_id")
    if (
        stored_session_id is not None
        and requested_session_id is not None
        and stored_session_id != requested_session_id
    ):
        raise ValueError(f"Task {task_id} is already bound to a different session.")
    return stored_session_id if stored_session_id is not None else requested_session_id


def copy_task_query(query: TaskQuery | None) -> TaskQuery:
    if query is None:
        return TaskQuery()
    if type(query) is not TaskQuery:
        raise TypeError("Task queries must be TaskQuery instances.")
    return TaskQuery(
        q=query.q,
        status=query.status,
        type=query.type,
        session_id=query.session_id,
        parent_task_id=query.parent_task_id,
        assigned_agent_name=query.assigned_agent_name,
        limit=query.limit,
        offset=query.offset,
        order_by=query.order_by,
    )


def copy_task_aggregate_filter(
    filters: TaskAggregateFilter | None,
) -> TaskAggregateFilter:
    if filters is None:
        return TaskAggregateFilter()
    if type(filters) is not TaskAggregateFilter:
        raise TypeError("Task aggregate filters must be TaskAggregateFilter instances.")
    return TaskAggregateFilter.model_validate(filters.model_dump(mode="python"))


def task_query_from_aggregate_filter(filters: TaskAggregateFilter) -> TaskQuery:
    filters = copy_task_aggregate_filter(filters)
    return TaskQuery(
        type=filters.type,
        session_id=filters.session_id,
        parent_task_id=filters.parent_task_id,
        assigned_agent_name=filters.assigned_agent_name,
    )


def require_contract_bound_task_creation_snapshot(task: Task) -> None:
    """Enforce the initial-snapshot reserve shared by every supporting store."""

    if type(task) is not Task or task.work_contract is None:
        raise TypeError("Creation-snapshot validation requires a contract-bound Task.")
    require_bounded_work_completion_document(
        task.model_dump(mode="json", warnings=False),
        "Contract-bound task creation snapshot",
        max_bytes=WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
        max_items=WORK_CONTRACT_TASK_CREATION_MAX_ITEMS,
    )


def _task_lifecycle_now(task: Task) -> datetime:
    """Return a wall-clock lifecycle time that cannot move ``task`` backward."""

    timestamps = [datetime.now(UTC), task.created_at, task.updated_at]
    if task.started_at is not None:
        timestamps.append(task.started_at)
    if task.completed_at is not None:
        timestamps.append(task.completed_at)
    return max(timestamps)


def _task_from_create(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding | None = None,
    retry_started_at: datetime | None = None,
    supports_verified_work_contracts: bool = False,
) -> Task:
    if request.work_contract is not None and not supports_verified_work_contracts:
        raise NotImplementedError(
            "This TaskStore does not support verified work-contract task bindings."
        )
    now = datetime.now(UTC)
    retry_started_at = (
        now
        if retry_started_at is None
        else normalize_utc_datetime(retry_started_at, "retry_started_at")
    )
    invocation = task_invocation_for_create(
        request,
        task_id=task_id,
        parent_task=parent_task,
        session_invocation=session_invocation,
    )
    retry_policy = request.retry_policy
    if retry_policy is None:
        retry_series = None
    else:
        retry_series_id = _task_retry_series_id(task_id)
        authority_sha256 = _task_retry_attempt_authority_sha256(
            task_id=task_id,
            task_type=request.type,
            title=request.title,
            description=request.description,
            parent_task_id=request.parent_task_id,
            assigned_agent_name=request.assigned_agent_name,
            available_at=request.available_at,
            created_at=now,
            task_input=request.input,
            metadata=request.metadata,
            invocation=invocation,
            series_id=retry_series_id,
            causal_budget_id=retry_series_id,
            attempt=1,
            policy=retry_policy,
            started_at=retry_started_at,
            cumulative_tokens=0,
            cumulative_estimated_cost=Decimal(0),
            predecessor_task_id=None,
        )
        retry_series = _task_retry_series_snapshot(
            series_id=retry_series_id,
            causal_budget_id=retry_series_id,
            authority_sha256=authority_sha256,
            attempt=1,
            policy=retry_policy,
            started_at=retry_started_at,
        )
    task = Task(
        id=task_id,
        type=request.type,
        title=request.title,
        description=request.description,
        status=TaskStatus.PENDING,
        session_id=request.session_id,
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        available_at=request.available_at,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        created_at=now,
        updated_at=now,
        invocation=invocation,
        retry_series=retry_series,
        work_contract=copy_work_contract_ref(request.work_contract),
    )
    if task.work_contract is not None:
        require_contract_bound_task_creation_snapshot(task)
    return task


def _running_task_from_create(
    request: TaskCreate,
    *,
    task_id: str,
    parent_task: Task | TaskInvocationSnapshot | None,
    session_invocation: SessionInvocationBinding,
    retry_started_at: datetime | None = None,
    supports_verified_work_contracts: bool = False,
) -> Task:
    task = _task_from_create(
        request,
        task_id=task_id,
        parent_task=parent_task,
        session_invocation=session_invocation,
        retry_started_at=retry_started_at,
        supports_verified_work_contracts=supports_verified_work_contracts,
    )
    if task.session_id is None:
        raise ValueError("TaskCreate.session_id is required to create a running task.")
    running = task.model_copy(
        update={
            "status": TaskStatus.RUNNING,
            "started_at": task.created_at,
        }
    )
    return copy_task(running) if running.work_contract is not None else running


def preflight_contract_bound_task_creation(
    request: TaskCreate,
    *,
    parent_task: Task | TaskInvocationSnapshot | None,
) -> None:
    """Validate the authoritative pending snapshot before an extension mutates."""

    if type(request) is not TaskCreate or request.work_contract is None:
        raise TypeError("Creation preflight requires a contract-bound TaskCreate request.")
    if request.task_id is None:
        raise ValueError("Contract-bound task creation requires a caller-stable task_id.")
    preview = _task_from_create(
        request,
        task_id=request.task_id,
        parent_task=parent_task,
        supports_verified_work_contracts=True,
    )
    del preview


def _ensure_can_transition(task: Task, next_status: TaskStatus) -> None:
    _ensure_task_status_can_transition(task.id, task.status, next_status)


def _ensure_retry_series_queue_attempt(
    retry_series: TaskRetrySeriesSnapshot | None,
) -> None:
    if retry_series is not None:
        raise ValueError(
            "Retry-series attempts are settled by task workers and cannot attach to sessions."
        )


def _ensure_task_status_can_transition(
    task_id: str,
    status: TaskStatus,
    next_status: TaskStatus,
) -> None:
    if status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        raise ValueError(f"Task {task_id} is already terminal: {status}")
    if next_status == TaskStatus.RUNNING and status != TaskStatus.PENDING:
        raise ValueError(f"Task {task_id} cannot transition to running from {status}")


def _ensure_can_hold_task(task: Task, next_status: TaskStatus) -> None:
    if next_status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} cannot be held as {next_status}.")
    _ensure_not_terminal(task)
    if _task_retry_cancellation_requested(task):
        raise TaskTerminalizationConflict(
            "Task retry cancellation is still draining under its current owner."
        )
    if task.status is TaskStatus.RUNNING and task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    if task.status not in {
        TaskStatus.PENDING,
        TaskStatus.CLAIMED,
        TaskStatus.RUNNING,
        *_HELD_TASK_STATUSES,
    }:
        raise ValueError(f"Task {task.id} cannot transition to {next_status} from {task.status}")


def _ensure_can_resume_task(task: Task) -> None:
    _ensure_not_terminal(task)
    if task.status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is not paused, blocked, or waiting for attention.")


def _ensure_not_terminal(task: Task) -> None:
    if task.status in _TERMINAL_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")


def _can_attach_claimed_task(
    task: Task,
    *,
    worker_id: str,
    now: datetime | None = None,
) -> bool:
    now = datetime.now(UTC) if now is None else now
    return _can_attach_claimed_task_state(
        status=task.status,
        session_id=task.session_id,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        expected_worker_id=worker_id,
        now=now,
    )


def _can_attach_claimed_task_state(
    *,
    status: TaskStatus,
    session_id: str | None,
    worker_id: str | None,
    lease_expires_at: datetime | None,
    expected_worker_id: str,
    now: datetime,
) -> bool:
    return (
        status is TaskStatus.CLAIMED
        and worker_id == expected_worker_id
        and session_id is None
        and lease_expires_at is not None
        and lease_expires_at > now
    )


def _ensure_active_task_lease(task: Task, worker_id: str, *, now: datetime | None = None) -> None:
    now = datetime.now(UTC) if now is None else now
    if task.lease_expires_at is None:
        raise TaskClaimLost(f"Task {task.id} has no active lease.")
    if task.lease_expires_at <= now:
        raise TaskClaimLost(f"Task {task.id} lease for worker {worker_id} has expired.")


def _ensure_owned_active_task_lease(
    task: Task,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Require the supplied worker to own the task's current live lease."""

    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed or running.")
    if task.worker_id != worker_id:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    _ensure_active_task_lease(task, worker_id, now=now)


def _raise_task_claim_attach_error(
    task: Task,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> None:
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed by worker {worker_id}.")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.status is TaskStatus.RUNNING:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        raise ValueError(f"Task {task.id} is already running.")
    if task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    raise RuntimeError(f"Task {task.id} active claim could not be attached.")


def _task_matches(task: Task, query: TaskQuery) -> bool:
    if query.q is not None and not _task_matches_search(task, query.q):
        return False
    if query.status is not None and task.status != query.status:
        return False
    if query.type is not None and task.type != query.type:
        return False
    if query.session_id is not None and task.session_id != query.session_id:
        return False
    if query.parent_task_id is not None and task.parent_task_id != query.parent_task_id:
        return False
    return not (
        query.assigned_agent_name is not None
        and task.assigned_agent_name != query.assigned_agent_name
    )


def _task_matches_search(task: Task, query: str) -> bool:
    needle = query.casefold()
    haystacks = (
        task.id,
        task.type,
        task.title,
        task.description,
        task.status.value,
        task.session_id,
        task.parent_task_id,
        task.assigned_agent_name,
        task.worker_id,
        task.status_reason,
    )
    return any(value is not None and needle in value.casefold() for value in haystacks)


def _task_matches_claim_filter(task: Task, query: TaskQuery) -> bool:
    if query.type is not None and task.type != query.type:
        return False
    if query.parent_task_id is not None and task.parent_task_id != query.parent_task_id:
        return False
    return not (
        query.assigned_agent_name is not None
        and task.assigned_agent_name != query.assigned_agent_name
    )


def _ensure_claim_query_supported(query: TaskQuery) -> None:
    if query.q is not None:
        raise ValueError("Task claim queries do not support q.")
    if query.session_id is not None:
        raise ValueError("Task claim queries do not support session_id.")
    if query.limit != TaskQuery.model_fields["limit"].default:
        raise ValueError("Task claim queries do not support limit.")
    if query.offset != TaskQuery.model_fields["offset"].default:
        raise ValueError("Task claim queries do not support offset.")


def _sort_tasks(tasks: list[Task], order_by: TaskOrder) -> list[Task]:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return sorted(tasks, key=lambda task: (task.created_at, task.id))
    if order_by == TaskOrder.CREATED_AT_DESC:
        return sorted(
            sorted(tasks, key=lambda task: task.id),
            key=lambda task: task.created_at,
            reverse=True,
        )
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return sorted(tasks, key=lambda task: (task.updated_at, task.id))
    return sorted(
        sorted(tasks, key=lambda task: task.id),
        key=lambda task: task.updated_at,
        reverse=True,
    )


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value


def _copy_optional_status_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return require_nonblank(value, "reason")


def _copy_optional_status_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return copy_durable_json_object(value, "payload")


_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}

_HELD_TASK_STATUSES = {
    TaskStatus.PAUSED,
    TaskStatus.BLOCKED,
    TaskStatus.NEEDS_ATTENTION,
}
