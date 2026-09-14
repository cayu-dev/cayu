"""Bounded one-shot task scheduling contracts and store-clock decisions.

Scheduling is eligibility authority, not worker ownership. These contracts do
not create a timer, dispatch work, or stop a claimed operation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    require_durable_clean_nonblank,
    revalidate_model_input,
)

TASK_SCHEDULE_ID_MAX_BYTES = 256
TASK_SCHEDULE_GRACE_MAX_SECONDS = 366 * 24 * 60 * 60


class TaskScheduleConflict(ValueError):
    """A scheduling operation does not match the current or retained authority."""


class TaskMisfirePolicy(StrEnum):
    FIRE_ONCE = "fire_once"
    SKIP = "skip"


class TaskScheduleEligibility(StrEnum):
    FUTURE = "future"
    ELIGIBLE = "eligible"
    MISFIRED = "misfired"
    EXPIRED = "expired"
    SKIPPED = "skipped"


class TaskSchedulePolicy(BaseModel):
    """An optional latest admission time and deterministic late-arrival policy.

    ``expires_at`` is exclusive. A task observed exactly at expiry cannot be
    admitted. Lateness strictly greater than the grace interval is a misfire;
    equality stays eligible. Expiry takes precedence over the misfire policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    expires_at: datetime | None = None
    misfire_policy: TaskMisfirePolicy = TaskMisfirePolicy.FIRE_ONCE
    misfire_grace_seconds: StrictInt = Field(default=60, ge=0, le=TASK_SCHEDULE_GRACE_MAX_SECONDS)

    @field_validator("expires_at")
    @classmethod
    def normalize_expiry(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "expires_at")


def copy_task_schedule_policy(value: TaskSchedulePolicy) -> TaskSchedulePolicy:
    """Reconstruct caller-owned fields without serializing an unvalidated model."""

    if type(value) is not TaskSchedulePolicy:
        raise TypeError("A typed task schedule policy is required.")
    return TaskSchedulePolicy.model_validate(
        {
            "expires_at": value.expires_at,
            "misfire_policy": value.misfire_policy,
            "misfire_grace_seconds": value.misfire_grace_seconds,
        }
    )


def validate_task_schedule_window(
    available_at: datetime, policy: TaskSchedulePolicy
) -> tuple[datetime, TaskSchedulePolicy]:
    available_at = normalize_utc_datetime(available_at, "available_at")
    policy = copy_task_schedule_policy(policy)
    if policy.expires_at is not None and policy.expires_at <= available_at:
        raise ValueError("Task schedule expiry must be later than availability.")
    return available_at, policy


def task_schedule_eligibility(
    *, available_at: datetime, policy: TaskSchedulePolicy, as_of: datetime
) -> TaskScheduleEligibility:
    """Classify an unadmitted schedule using one store-owned clock snapshot."""

    available_at, policy = validate_task_schedule_window(available_at, policy)
    as_of = normalize_utc_datetime(as_of, "as_of")
    if as_of < available_at:
        return TaskScheduleEligibility.FUTURE
    if policy.expires_at is not None and as_of >= policy.expires_at:
        return TaskScheduleEligibility.EXPIRED
    # Compare elapsed seconds rather than adding a duration to available_at:
    # valid timestamps near datetime.max must not overflow during admission.
    if (as_of - available_at).total_seconds() > policy.misfire_grace_seconds:
        return (
            TaskScheduleEligibility.SKIPPED
            if policy.misfire_policy is TaskMisfirePolicy.SKIP
            else TaskScheduleEligibility.MISFIRED
        )
    return TaskScheduleEligibility.ELIGIBLE


def _schedule_identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TASK_SCHEDULE_ID_MAX_BYTES:
        raise ValueError(f"{field_name} exceeds the task schedule identity limit.")
    return value


class TaskRescheduleRequest(BaseModel):
    """Replace a not-yet-admitted schedule at one expected revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    operation_id: str
    expected_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    available_at: datetime
    policy: TaskSchedulePolicy = Field(default_factory=TaskSchedulePolicy)

    @field_validator("task_id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _schedule_identity(value, info.field_name)

    @field_validator("available_at")
    @classmethod
    def normalize_availability(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "available_at")

    @field_validator("policy", mode="before")
    @classmethod
    def copy_policy(cls, value: object) -> object:
        if type(value) is TaskSchedulePolicy:
            return copy_task_schedule_policy(value)
        return revalidate_model_input(value, TaskSchedulePolicy)

    @model_validator(mode="after")
    def validate_window(self) -> TaskRescheduleRequest:
        validate_task_schedule_window(self.available_at, self.policy)
        return self


class TaskScheduleCancelRequest(BaseModel):
    """Cancel one exact schedule revision without bypassing a live worker."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    operation_id: str
    expected_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    @field_validator("task_id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _schedule_identity(value, info.field_name)


class TaskScheduleWakeup(BaseModel):
    """A bounded, content-free store observation, never permission to dispatch.

    The store applies the worker's actual claim filters. ``next_available_at``
    is absent when no matching unadmitted future task exists. Expired/skipped
    work may require a sweep even when no task will become runnable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    as_of: datetime
    next_available_at: datetime | None = None
    next_expiry_at: datetime | None = None
    maintenance_required: StrictBool = False

    @field_validator("as_of", "next_available_at", "next_expiry_at")
    @classmethod
    def normalize_time(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)


class TaskScheduleState(BaseModel):
    """Current schedule authority embedded in the durable task snapshot.

    The original creation digest survives rescheduling and terminalization.
    First admission permanently ends eligibility expiry: lease reclaim does
    not turn an already-dispatched task into a new scheduled occurrence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    revision: StrictInt = Field(default=1, ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    policy: TaskSchedulePolicy
    creation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admitted_at: datetime | None = None

    @field_validator("policy", mode="before")
    @classmethod
    def copy_policy(cls, value: object) -> object:
        return revalidate_model_input(value, TaskSchedulePolicy)

    @field_validator("admitted_at")
    @classmethod
    def normalize_admission(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "admitted_at")


class TaskScheduleEventType(StrEnum):
    SCHEDULED = "task.scheduled"
    RESCHEDULED = "task.rescheduled"
    ELIGIBLE = "task.schedule_eligible"
    MISFIRED = "task.schedule_misfired"
    EXPIRED = "task.schedule_expired"
    SKIPPED = "task.schedule_skipped"
    CLAIMED = "task.schedule_claimed"
    CANCELLATION_REQUESTED = "task.schedule_cancellation_requested"
    CANCELLED = "task.schedule_cancelled"
    STARTED = "task.schedule_started"
    HELD = "task.schedule_held"
    RESUMED = "task.schedule_resumed"
    COMPLETED = "task.schedule_completed"
    FAILED = "task.schedule_failed"


class TaskScheduleEvent(BaseModel):
    """Task-owned evidence; it exists even before a session is attached."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    type: TaskScheduleEventType
    revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    occurred_at: datetime
    available_at: datetime
    policy: TaskSchedulePolicy
    invocation_id: str
    operation_id: str | None = None

    @field_validator("task_id", "invocation_id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _schedule_identity(value, info.field_name)

    @field_validator("occurred_at", "available_at")
    @classmethod
    def normalize_time(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("policy", mode="before")
    @classmethod
    def copy_policy(cls, value: object) -> object:
        return revalidate_model_input(value, TaskSchedulePolicy)


class TaskScheduleReceipt(BaseModel):
    """Immutable accepted mutation; replay never overwrites the live schedule."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    operation_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    schedule: TaskScheduleState
    available_at: datetime
    committed_at: datetime
    type: Literal[
        TaskScheduleEventType.RESCHEDULED,
        TaskScheduleEventType.CANCELLATION_REQUESTED,
        TaskScheduleEventType.CANCELLED,
    ]

    @field_validator("task_id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _schedule_identity(value, info.field_name)

    @field_validator("available_at", "committed_at")
    @classmethod
    def normalize_time(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("schedule", mode="before")
    @classmethod
    def copy_schedule(cls, value: object) -> object:
        return revalidate_model_input(value, TaskScheduleState)

    @model_validator(mode="after")
    def validate_successor(self) -> TaskScheduleReceipt:
        if self.schedule.revision != self.expected_revision + 1:
            raise ValueError("Task schedule receipt does not advance the expected revision.")
        validate_task_schedule_window(self.available_at, self.schedule.policy)
        return self
