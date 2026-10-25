"""Retry budgets: a cap on how many tasks one retry key may create.

An application that retries an event by creating a new task for it (a sweeper,
a reconciler, a redelivered webhook) passes the same ``TaskCreate.retry_key``
each time. The task store counts the tasks created for that key and refuses a
new one once the key's ``TaskRetryBudget`` is spent, recording the key as given
up. That bounds an event-level retry loop that a per-task time limit cannot.

This is separate from ``TaskRetryPolicy``, where Cayu itself creates each
successor of a failed attempt. A retry budget never creates tasks; it only
admits or refuses the ones the application creates. The two can be combined:
each admitted task may still have its own retry series.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

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
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.tasks.records import TASK_RETRY_MAX_ATTEMPTS, TaskRetrySeriesDisposition

TASK_RETRY_KEY_MAX_BYTES = 256

__all__ = [
    "DEFAULT_TASK_RETRY_BUDGET",
    "TASK_RETRY_KEY_MAX_BYTES",
    "TaskRetryBudget",
    "TaskRetryBudgetExhausted",
    "TaskRetryKeyState",
    "TaskRetryKeyStatus",
    "admit_task_retry_key",
]


class TaskRetryBudget(BaseModel):
    """How many tasks one retry key may create, and for how long.

    ``max_attempts`` counts tasks created with the key, the first included.
    ``max_elapsed_seconds`` is measured from the first one; after it, no new
    task is admitted even if attempts remain. A task already admitted keeps
    running. ``None`` turns either limit off, and
    :meth:`TaskRetryBudget.unlimited` turns both off: the store still counts
    attempts but never refuses one.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )

    max_attempts: StrictInt | None = Field(default=3, ge=1, le=TASK_RETRY_MAX_ATTEMPTS)
    max_elapsed_seconds: StrictFloat | StrictInt | None = Field(default=1800.0, gt=0, le=31_536_000)

    @field_validator("max_elapsed_seconds")
    @classmethod
    def as_float(cls, value: float | int | None) -> float | None:
        return None if value is None else float(value)

    @classmethod
    def unlimited(cls) -> TaskRetryBudget:
        """A budget that counts attempts but never refuses one."""
        return cls(max_attempts=None, max_elapsed_seconds=None)


#: Applied when a ``TaskCreate`` sets ``retry_key`` but no ``retry_budget``.
DEFAULT_TASK_RETRY_BUDGET = TaskRetryBudget()


class TaskRetryKeyStatus(StrEnum):
    ACTIVE = "active"
    GIVEN_UP = "given_up"


class TaskRetryKeyState(BaseModel):
    """Durable state of one retry key. ``given_up`` is terminal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    retry_key: str
    status: TaskRetryKeyStatus = TaskRetryKeyStatus.ACTIVE
    attempts: StrictInt = Field(ge=1)
    budget: TaskRetryBudget
    first_attempt_at: datetime
    latest_attempt_at: datetime
    latest_task_id: str
    given_up_at: datetime | None = None
    given_up_reason: TaskRetrySeriesDisposition | None = None
    refused_task_id: str | None = None

    @field_validator("retry_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_task_retry_key(value)

    @field_validator("latest_task_id", "refused_task_id")
    @classmethod
    def validate_task_id(cls, value: str | None, info) -> str | None:
        return None if value is None else require_clean_nonblank(value, info.field_name)

    @field_validator("first_attempt_at", "latest_attempt_at", "given_up_at")
    @classmethod
    def normalize_time(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, info.field_name)

    @field_validator("given_up_reason")
    @classmethod
    def validate_reason(
        cls, value: TaskRetrySeriesDisposition | None
    ) -> TaskRetrySeriesDisposition | None:
        if value not in (
            None,
            TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED,
            TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        ):
            raise ValueError("A retry key gives up only when attempts or time run out.")
        return value

    @model_validator(mode="after")
    def validate_given_up_evidence(self) -> TaskRetryKeyState:
        evidence = {
            item is not None
            for item in (self.given_up_at, self.given_up_reason, self.refused_task_id)
        }
        if evidence != {self.status is TaskRetryKeyStatus.GIVEN_UP}:
            raise ValueError("Only a given-up retry key records when and why it gave up.")
        return self

    @property
    def attempts_remaining(self) -> int | None:
        """Attempts the current budget still admits, or ``None`` without a cap."""
        if self.status is TaskRetryKeyStatus.GIVEN_UP:
            return 0
        if self.budget.max_attempts is None:
            return None
        return max(0, self.budget.max_attempts - self.attempts)

    @property
    def elapsed_deadline(self) -> datetime | None:
        """When the current budget stops admitting attempts, or ``None``."""
        if self.budget.max_elapsed_seconds is None:
            return None
        return self.first_attempt_at + timedelta(seconds=self.budget.max_elapsed_seconds)

    def inspection(self) -> dict[str, Any]:
        """A JSON-safe view with the derived fields, for logs and ``cayu inspect``."""
        document = self.model_dump(mode="json")
        document["attempts_remaining"] = self.attempts_remaining
        deadline = self.elapsed_deadline
        document["elapsed_deadline"] = None if deadline is None else deadline.isoformat()
        return document


class TaskRetryBudgetExhausted(ValueError):
    """A retry key has used its budget; the store created no task.

    ``state`` is the key's durable, given-up state. Every later task for the
    same key is refused the same way, whatever budget it carries.
    """

    def __init__(self, state: TaskRetryKeyState) -> None:
        self.state = state
        reason = "" if state.given_up_reason is None else f" ({state.given_up_reason.value})"
        super().__init__(
            f"Retry key {state.retry_key!r} gave up after {state.attempts} "
            f"attempt{'s' if state.attempts != 1 else ''}{reason}; no task was created."
        )


def validate_task_retry_key(value: str) -> str:
    value = require_clean_nonblank(value, "retry_key")
    if len(value.encode("utf-8")) > TASK_RETRY_KEY_MAX_BYTES:
        raise ValueError(f"retry_key must be at most {TASK_RETRY_KEY_MAX_BYTES} UTF-8 bytes.")
    return value


def copy_task_retry_budget(budget: TaskRetryBudget | None) -> TaskRetryBudget | None:
    if budget is None:
        return None
    if type(budget) is not TaskRetryBudget:
        raise TypeError("retry_budget must be a TaskRetryBudget.")
    return TaskRetryBudget(
        max_attempts=budget.max_attempts, max_elapsed_seconds=budget.max_elapsed_seconds
    )


def stored_task_retry_key_state(document: str, retry_key: str) -> TaskRetryKeyState:
    """Decode a stored key state and check it belongs to the row's key."""
    state = TaskRetryKeyState.model_validate_json(document)
    if state.retry_key != retry_key:
        raise RuntimeError("Stored task retry key state conflicts with its key.")
    return state


def task_retry_key_request(request: Any) -> tuple[str, TaskRetryBudget] | None:
    """The key and effective budget a ``TaskCreate`` asks a store to admit."""
    if request.retry_key is None:
        return None
    return request.retry_key, request.retry_budget or DEFAULT_TASK_RETRY_BUDGET


def admit_task_retry_key(
    state: TaskRetryKeyState | None,
    *,
    retry_key: str,
    budget: TaskRetryBudget,
    task_id: str,
    now: datetime,
) -> tuple[TaskRetryKeyState, bool]:
    """Decide one new task for ``retry_key`` from store-authoritative ``now``.

    Returns the state to persist and whether the task is admitted. A store calls
    this inside the transaction that inserts the task, after it has ruled out a
    duplicate task ID, and persists the state even when the task is refused.
    The budget is the one on the new request, so raising it in code applies to
    keys that are still active; a key that has given up stays given up.
    """
    now = normalize_utc_datetime(now, "now")
    if state is None:
        return (
            TaskRetryKeyState(
                retry_key=retry_key,
                attempts=1,
                budget=budget,
                first_attempt_at=now,
                latest_attempt_at=now,
                latest_task_id=task_id,
            ),
            True,
        )
    if state.retry_key != retry_key:
        raise AssertionError("Retry key state belongs to another key.")
    if state.status is TaskRetryKeyStatus.GIVEN_UP:
        return state, False
    reason: TaskRetrySeriesDisposition | None = None
    if budget.max_attempts is not None and state.attempts >= budget.max_attempts:
        reason = TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
    elif budget.max_elapsed_seconds is not None and now >= state.first_attempt_at + timedelta(
        seconds=budget.max_elapsed_seconds
    ):
        reason = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
    if reason is not None:
        return (
            state.model_copy(
                update={
                    "status": TaskRetryKeyStatus.GIVEN_UP,
                    "budget": budget,
                    "given_up_at": now,
                    "given_up_reason": reason,
                    "refused_task_id": task_id,
                }
            ),
            False,
        )
    return (
        state.model_copy(
            update={
                "attempts": state.attempts + 1,
                "budget": budget,
                "latest_attempt_at": now,
                "latest_task_id": task_id,
            }
        ),
        True,
    )


async def unscoped_task_retry_key(retry_key: str) -> str:
    """Validate a key for reading; keys are store-wide, so scoped readers are refused."""
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.resource_access import current_data_bounds

    retry_key = validate_task_retry_key(retry_key)
    if await current_data_bounds("tasks") is not None:
        raise ResourceAccessDenied()
    return retry_key
