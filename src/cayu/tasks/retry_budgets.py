"""Event identity, atomic attempt admission, and shared execution deadlines.

The first task fixes an event's budget. All successor tasks share its count and
absolute deadline. Workers cancel expired work while retaining ownership until
cleanup finishes; terminal task transitions durably settle the event.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, NoReturn

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
from cayu._validation import canonical_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.tasks.records import TaskRetrySeriesDisposition, TaskStatus

TASK_RETRY_KEY_MAX_BYTES = 256
_TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})

__all__ = [
    "DEFAULT_TASK_RETRY_BUDGET",
    "TASK_RETRY_KEY_MAX_BYTES",
    "TaskEventAlreadyCompleted",
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
    task is admitted even if attempts remain, and workers cancel admitted work.
    ``None`` turns either limit off; :meth:`TaskRetryBudget.unlimited` turns
    both off. A successfully completed event still refuses duplicate attempts.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )

    max_attempts: StrictInt | None = Field(default=3, ge=1)
    max_elapsed_seconds: StrictFloat | StrictInt | None = Field(default=1800.0, gt=0)

    @field_validator("max_elapsed_seconds")
    @classmethod
    def as_float(cls, value: float | int | None) -> float | None:
        return None if value is None else float(value)

    @classmethod
    def unlimited(cls) -> TaskRetryBudget:
        """A budget without attempt or elapsed limits."""
        return cls(max_attempts=None, max_elapsed_seconds=None)


#: Applied to ordinary root events without an explicit budget or retry policy.
DEFAULT_TASK_RETRY_BUDGET = TaskRetryBudget()


class TaskRetryKeyStatus(StrEnum):
    ACTIVE = "active"
    GIVEN_UP = "given_up"
    SUCCEEDED = "succeeded"


class TaskRetryKeyState(BaseModel):
    """Durable state of one retry key. ``given_up`` is terminal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    retry_key: str
    status: TaskRetryKeyStatus = TaskRetryKeyStatus.ACTIVE
    attempts: StrictInt = Field(ge=1)
    settled_attempts: StrictInt = Field(default=0, ge=0)
    legacy_attempts: StrictInt = Field(default=0, ge=0)
    succeeded_at: datetime | None = None
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

    @field_validator("first_attempt_at", "latest_attempt_at", "given_up_at", "succeeded_at")
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
        given_up = self.status is TaskRetryKeyStatus.GIVEN_UP
        if given_up != (self.given_up_at is not None and self.given_up_reason is not None):
            raise ValueError("Only a given-up retry key records when and why it gave up.")
        if not given_up and self.refused_task_id is not None:
            raise ValueError("Only a given-up retry key records a refused task.")
        if (self.status is TaskRetryKeyStatus.SUCCEEDED) != (self.succeeded_at is not None):
            raise ValueError("Only a successful event records success time.")
        if self.settled_attempts + self.legacy_attempts > self.attempts:
            raise ValueError("An event cannot settle more attempts than it admitted.")
        return self

    @property
    def attempts_remaining(self) -> int | None:
        """Attempts the current budget still admits, or ``None`` without a cap."""
        if self.status is not TaskRetryKeyStatus.ACTIVE:
            return 0
        if self.budget.max_attempts is None:
            return None
        return max(0, self.budget.max_attempts - self.attempts)

    @property
    def elapsed_deadline(self) -> datetime | None:
        """When the current budget stops admitting attempts, or ``None``."""
        if self.budget.max_elapsed_seconds is None:
            return None
        try:
            return self.first_attempt_at + timedelta(seconds=self.budget.max_elapsed_seconds)
        except OverflowError as exc:
            raise ValueError(
                "The event deadline is outside the representable UTC datetime range."
            ) from exc

    def inspection(self) -> dict[str, Any]:
        """A JSON-safe view with the derived fields, for logs and ``cayu inspect``."""
        document = self.model_dump(mode="json")
        document["attempts_remaining"] = self.attempts_remaining
        deadline = self.elapsed_deadline
        document["elapsed_deadline"] = None if deadline is None else deadline.isoformat()
        return document


class TaskRetryBudgetExhausted(ValueError):
    """A retry key has used its budget; the store created no task.

    ``state`` records the durable event policy and counts. It may remain active
    while already-admitted work finishes. Given-up events cannot be reopened.
    """

    def __init__(self, state: TaskRetryKeyState) -> None:
        self.state = state
        reason = "" if state.given_up_reason is None else f" ({state.given_up_reason.value})"
        super().__init__(
            f"Retry key {state.retry_key!r} exhausted admission after {state.attempts} "
            f"attempt{'s' if state.attempts != 1 else ''}{reason}; no task was created."
        )


class TaskEventAlreadyCompleted(ValueError):
    """The event already succeeded; a new attempt would duplicate completed work."""

    def __init__(self, state: TaskRetryKeyState) -> None:
        self.state = state
        super().__init__(f"Event {state.retry_key!r} already completed; no task was created.")


def raise_task_retry_key_closed(state: TaskRetryKeyState) -> NoReturn:
    if state.status is TaskRetryKeyStatus.SUCCEEDED:
        raise TaskEventAlreadyCompleted(state)
    raise TaskRetryBudgetExhausted(state)


def expire_task_retry_key(state: TaskRetryKeyState, *, now: datetime) -> TaskRetryKeyState:
    if (
        state.status is TaskRetryKeyStatus.ACTIVE
        and state.elapsed_deadline is not None
        and now >= state.elapsed_deadline
    ):
        return state.model_copy(
            update={
                "status": TaskRetryKeyStatus.GIVEN_UP,
                "given_up_at": now,
                "given_up_reason": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
            }
        )
    return state


def observe_task_retry_key_transition(
    state: TaskRetryKeyState,
    prior: Any,
    current: Any,
    *,
    now: datetime,
) -> TaskRetryKeyState:
    """Account a committed task transition once, under the key's transaction lock."""

    state = expire_task_retry_key(state, now=now)
    terminal = _TERMINAL_TASK_STATUSES
    if prior is None and current.id != state.latest_task_id:
        # Native retry-series successors retain the immutable event association.
        state, admitted = admit_task_retry_key(
            state,
            retry_key=state.retry_key,
            budget=state.budget,
            task_id=current.id,
            now=now,
        )
        if not admitted:
            raise RuntimeError("Retry settlement admitted a successor after its event closed.")
    if (prior is None or prior.status not in terminal) and current.status in terminal:
        state = state.model_copy(update={"settled_attempts": state.settled_attempts + 1})
        if state.status is TaskRetryKeyStatus.ACTIVE:
            if current.status is TaskStatus.COMPLETED:
                return state.model_copy(
                    update={
                        "status": TaskRetryKeyStatus.SUCCEEDED,
                        "succeeded_at": now,
                    }
                )
            if (
                state.budget.max_attempts is not None
                and state.attempts >= state.budget.max_attempts
                and state.settled_attempts + state.legacy_attempts == state.attempts
            ):
                return state.model_copy(
                    update={
                        "status": TaskRetryKeyStatus.GIVEN_UP,
                        "given_up_at": now,
                        "given_up_reason": TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED,
                    }
                )
    return state


def task_event_retry_refusal(state: TaskRetryKeyState | None, *, now: datetime):
    """A successor's admission reason, without mutating the event before settlement."""
    if state is None:
        return None
    if state.status is TaskRetryKeyStatus.GIVEN_UP:
        return state.given_up_reason
    if state.status is TaskRetryKeyStatus.SUCCEEDED:
        return TaskRetrySeriesDisposition.CANCELLED
    if state.elapsed_deadline is not None and now >= state.elapsed_deadline:
        return TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
    if state.budget.max_attempts is not None and state.attempts >= state.budget.max_attempts:
        return TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
    return None


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
    raw = json.loads(document)
    if "settled_attempts" not in raw:
        # Historical attempts have no event association or terminal accounting.
        # Preserve their admission count without inventing settlement evidence.
        raw["legacy_attempts"] = raw["attempts"]
    state = TaskRetryKeyState.model_validate(raw)
    if state.retry_key != retry_key:
        raise RuntimeError("Stored task retry key state conflicts with its key.")
    return state


def task_event_applies(request: Any) -> bool:
    return request.retry_key is not None or all(
        value is None
        for value in (
            request.session_id,
            request.parent_task_id,
            request.work_contract,
            request.schedule_policy,
        )
    )


def task_event_key(request: Any, *, task_id: str, invocation: Any) -> str | None:
    """Resolve declared event identity, never a fingerprint of the job payload."""
    if request.retry_key is not None:
        return request.retry_key
    if not task_event_applies(request):
        return None
    event_id = request.event_id
    if event_id is None:
        supplied = request.input.get("event_id")
        event_id = supplied if isinstance(supplied, str) and supplied.strip() else None
    identity = {
        "type": request.type,
        "event_id": event_id,
        "independent_task_id": task_id if event_id is None else None,
        "origin": invocation.origin.model_dump(mode="json"),
        "resource_access": (
            None
            if invocation.resource_access is None
            else {
                "authority": invocation.resource_access.authority,
                "subject": invocation.resource_access.subject,
            }
        ),
        "access_labels": dict(invocation.access_labels),
    }
    return "event:" + sha256(canonical_durable_json_bytes(identity, "event identity")).hexdigest()


def task_retry_key_request(request: Any, task: Any = None) -> tuple[str, TaskRetryBudget] | None:
    """The persisted event key and effective budget a store must admit atomically."""
    key = request.retry_key if task is None else task.invocation.event_retry_key
    if key is None:
        return None
    budget = request.retry_budget
    if budget is None and request.retry_policy is not None:
        budget = TaskRetryBudget(
            max_attempts=request.retry_policy.max_attempts,
            max_elapsed_seconds=request.retry_policy.max_elapsed_seconds,
        )
    return key, budget or DEFAULT_TASK_RETRY_BUDGET


def admit_task_retry_key(
    state: TaskRetryKeyState | None,
    *,
    retry_key: str,
    budget: TaskRetryBudget,
    task_id: str,
    now: datetime,
    budget_is_explicit: bool = True,
) -> tuple[TaskRetryKeyState, bool]:
    """Decide one new task for ``retry_key`` from store-authoritative ``now``.

    Returns the state to persist and whether the task is admitted. A store calls
    this inside the transaction that inserts the task, after it has ruled out a
    duplicate task ID, and persists the state even when the task is refused.
    The first admission fixes policy. Omitted budgets inherit it; an explicit
    budget must match. No request can reopen a terminal event.
    """
    now = normalize_utc_datetime(now, "now")
    if state is None:
        initial = TaskRetryKeyState(
            retry_key=retry_key,
            attempts=1,
            budget=budget,
            first_attempt_at=now,
            latest_attempt_at=now,
            latest_task_id=task_id,
        )
        # Refuse an unrepresentable deadline before any durable row is inserted.
        _ = initial.elapsed_deadline
        return initial, True
    if state.retry_key != retry_key:
        raise AssertionError("Retry key state belongs to another key.")
    if state.status is not TaskRetryKeyStatus.ACTIVE:
        return state, False
    if budget_is_explicit and budget != state.budget:
        raise ValueError("An event's retry budget is fixed by its first admission.")
    budget = state.budget
    reason: TaskRetrySeriesDisposition | None = None
    if budget.max_elapsed_seconds is not None and now >= state.first_attempt_at + timedelta(
        seconds=budget.max_elapsed_seconds
    ):
        reason = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
    elif budget.max_attempts is not None and state.attempts >= budget.max_attempts:
        if state.settled_attempts + state.legacy_attempts < state.attempts:
            # Admission exhaustion does not revoke already-admitted work.
            return state, False
        reason = TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
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


async def enforce_task_event_deadline(task_store: Any, task: Any) -> Any:
    """Request owned cancellation; terminal task evidence still requires cleanup."""
    from cayu.tasks.cancellation import (
        _task_cancellation_requested,
        _task_retry_cancellation_requested,
    )
    from cayu.tasks.scheduling import TaskScheduleCancelRequest, TaskScheduleConflict
    from cayu.tasks.terminalization import TaskTerminalizationConflict

    if _task_cancellation_requested(task) or _task_retry_cancellation_requested(task):
        return task
    if not task_store.supports_task_event_execution or task.invocation.event_retry_key is None:
        return task
    if task.status in _TERMINAL_TASK_STATUSES:
        return task
    state = await task_store.refresh_task_retry_key(task.invocation.event_retry_key)
    if state is None:
        raise RuntimeError("Task event association has no durable budget record.")
    if state.status is TaskRetryKeyStatus.ACTIVE:
        return task
    reason = (
        "event_completed"
        if state.status is TaskRetryKeyStatus.SUCCEEDED
        else ("event_" + state.given_up_reason.value)
    )
    logging.getLogger(__name__).warning(
        "Event execution budget closed; requesting cancellation (%s).",
        reason,
    )
    try:
        if task.schedule is not None:
            await task_store.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=task.id,
                    operation_id=f"event-expiry:{task.schedule.revision}",
                    expected_revision=task.schedule.revision,
                )
            )
            return await task_store.load_task(task.id)
        return await task_store.cancel_task(task.id, {"code": reason, "retry_key": state.retry_key})
    except TaskScheduleConflict:
        # A concurrent reschedule advances its fence. A later maintenance pass
        # retries against that revision; never cancel using stale authority.
        current = await task_store.load_task(task.id)
        if current is not None and current.schedule != task.schedule:
            return current
        raise
    except (TaskTerminalizationConflict, ValueError):
        current = await task_store.load_task(task.id)
        if current is not None and (
            current.status in _TERMINAL_TASK_STATUSES
            or _task_cancellation_requested(current)
            or _task_retry_cancellation_requested(current)
        ):
            return current
        raise


async def reconcile_task_event_deadlines(task_store: Any) -> None:
    """One maintenance batch; active workers enforce their own event independently."""
    if not task_store.supports_task_event_execution:
        return
    for task_id in await task_store.expired_task_event_ids():
        task = await task_store.load_task(task_id)
        if task is not None:
            await enforce_task_event_deadline(task_store, task)
