"""Durable task records and their validation, independent of storage backends."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
    revalidate_model_input,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.sessions.invocation import SessionInvocationBinding, TaskInvocation, copy_task_invocation
from cayu.tasks.contracts import (
    WORK_CONTRACT_TASK_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_ITEMS,
    WorkContractRef,
    copy_work_contract_ref,
    preflight_work_completion_document,
    require_bounded_work_completion_document,
    validate_work_completion_linked_id,
)
from cayu.tasks.scheduling import TaskScheduleState, validate_task_schedule_window

_TASK_RETRY_COST_MAX_DIGITS = 64
_TASK_RETRY_TOTAL_COST_MAX_DIGITS = 128
_TASK_RETRY_COST_MAX_DECIMAL_PLACES = 64
_TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES = 1024


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
    WAITING_DEPENDENCIES = "waiting_dependencies"
    WAITING_GROUP = "waiting_group"
    DEPENDENCY_SKIPPED = "dependency_skipped"
    CLAIMED = "claimed"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"
    FAILED = "failed"
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


def _validate_task_retry_reconciliation_identity(
    value: str,
    field_name: str,
    *,
    max_bytes: int = _TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES,
) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} must be at most {max_bytes} UTF-8 bytes.")
    return value


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
    session_instance_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    graph_id: str | None = Field(default=None, frozen=True)
    prerequisite_task_ids: tuple[str, ...] = Field(default=(), frozen=True)
    available_at: datetime | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    interrupted_handoff_id: str | None = None
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
    schedule: TaskScheduleState | None = None
    retry_series: TaskRetrySeriesSnapshot | None = None
    work_contract: WorkContractRef | None = Field(default=None, frozen=True)

    @model_validator(mode="after")
    def validate_graph_membership(self) -> Task:
        from cayu.tasks._graph_identity import TASK_GRAPH_MAX_NODES, graph_identifier

        if self.graph_id is None:
            if self.prerequisite_task_ids or self.status in {
                TaskStatus.WAITING_DEPENDENCIES,
                TaskStatus.WAITING_GROUP,
                TaskStatus.DEPENDENCY_SKIPPED,
            }:
                raise ValueError("Dependency state requires graph membership.")
            return self
        graph_identifier(self.graph_id)
        graph_identifier(self.id)
        if len(self.prerequisite_task_ids) > TASK_GRAPH_MAX_NODES:
            raise ValueError("Graph prerequisite count exceeds its bound.")
        dependencies = tuple(graph_identifier(identity) for identity in self.prerequisite_task_ids)
        if dependencies != tuple(sorted(set(dependencies))) or self.id in dependencies:
            raise ValueError("Graph prerequisites must be distinct ordered identities.")
        if (
            self.status in {TaskStatus.WAITING_DEPENDENCIES, TaskStatus.DEPENDENCY_SKIPPED}
            and not dependencies
        ):
            raise ValueError("Dependency state requires prerequisites.")
        return self

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
        if info.field_name == "metadata":
            return copy_durable_metadata(value)
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
        "interrupted_handoff_id",
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

    @field_validator("session_instance_id")
    @classmethod
    def validate_session_instance_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return SessionInvocationBinding.validate_session_instance_id(value)

    @field_validator("available_at")
    @classmethod
    def normalize_available_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "available_at")

    @field_validator("schedule", mode="before")
    @classmethod
    def copy_schedule(cls, value: object) -> object:
        return revalidate_model_input(value, TaskScheduleState)

    @model_validator(mode="after")
    def validate_schedule(self) -> Task:
        if self.schedule is not None:
            if self.available_at is None:
                raise ValueError("Managed task schedules require available_at.")
            if self.schedule.admitted_at is None:
                validate_task_schedule_window(self.available_at, self.schedule.policy)
        return self

    @field_validator("work_contract", mode="before")
    @classmethod
    def copy_work_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @model_validator(mode="after")
    def validate_retry_and_work_contract_authority(self) -> Task:
        if self.session_instance_id is not None and self.session_id is None:
            raise ValueError("Task session-instance authority requires a session_id.")
        if self.interrupted_handoff_id is not None and (
            self.status is not TaskStatus.RUNNING
            or self.session_id is None
            or self.session_instance_id is None
        ):
            raise ValueError("Interrupted-handoff lineage requires an attached running task.")
        if self.retry_series is not None:
            _validate_task_retry_reconciliation_identity(self.id, "id")
            if self.worker_id is not None:
                _validate_task_retry_reconciliation_identity(self.worker_id, "worker_id")
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
        session_instance_id=task.session_instance_id,
        parent_task_id=task.parent_task_id,
        graph_id=task.graph_id,
        prerequisite_task_ids=task.prerequisite_task_ids,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        schedule=task.schedule,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        interrupted_handoff_id=task.interrupted_handoff_id,
        status_reason=task.status_reason,
        status_payload=(
            None
            if task.status_payload is None
            else copy_durable_json_object(task.status_payload, "status_payload")
        ),
        input=copy_durable_json_object(task.input, "input"),
        result=(None if task.result is None else copy_durable_json_object(task.result, "result")),
        error=None if task.error is None else copy_durable_json_object(task.error, "error"),
        metadata=copy_durable_metadata(task.metadata),
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
