"""What a task store does with a stored task series; each store applies it in its own transaction."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from cayu._clock import normalize_utc_datetime
from cayu.tasks.creation import _task_from_create, copy_task_create
from cayu.tasks.records import _TERMINAL_TASK_STATUSES, Task
from cayu.tasks.series import (
    _CANCELLED_SKIP_MAX,
    TaskSeries,
    TaskSeriesState,
    occurrence_request,
    occurrence_task_id,
    planned_occurrence,
    task_occurrence,
)


def saved_series_state(
    existing: TaskSeriesState | None, series: TaskSeries, *, now: datetime
) -> TaskSeriesState:
    series = TaskSeries.model_validate(series.model_dump())
    now = normalize_utc_datetime(now, "now")
    if existing is None:
        return TaskSeriesState(series=series, revision=1, updated_at=now)
    return existing.model_copy(
        update={"series": series, "revision": existing.revision + 1, "updated_at": now}
    )


def paused_series_state(
    existing: TaskSeriesState, *, paused: bool, now: datetime
) -> TaskSeriesState:
    if existing.paused is paused:
        return existing
    return existing.model_copy(
        update={"paused": paused, "revision": existing.revision + 1, "updated_at": now}
    )


def needs_occurrence(state: TaskSeriesState, next_task: Task | None) -> bool:
    """Whether an active series has nothing queued: no next task, or one that already ended."""

    return not state.paused and (next_task is None or next_task.status in _TERMINAL_TASK_STATUSES)


def finished_occurrence(prior: Task | None, current: Task) -> tuple[str, datetime] | None:
    """The series and time of an occurrence this transition ended, or ``None``."""

    if prior is None or prior.status in _TERMINAL_TASK_STATUSES:
        return None
    if current.status not in _TERMINAL_TASK_STATUSES:
        return None
    return task_occurrence(current)


def advances_after(state: TaskSeriesState, finished: Task, next_task: Task | None) -> bool:
    """Whether the end of ``finished`` should queue the series' next occurrence.

    Only the occurrence the store queued last advances it (or any, once that one
    ended), so an older occurrence that ends late can't fork the series.
    """

    if state.paused:
        return False
    return state.next_task_id in (None, finished.id) or needs_occurrence(state, next_task)


def _taken(existing: Task | None) -> bool:
    # An occurrence that already ended can't advance the series, so it is stepped over.
    return existing is None or existing.status not in _TERMINAL_TASK_STATUSES


def _advanced(
    state: TaskSeriesState, at: datetime | None, existing: Task | None, now: datetime
) -> tuple[TaskSeriesState, Task | None]:
    if at is None:
        return state.model_copy(update={"next_task_id": None, "updated_at": now}), None
    task_id = occurrence_task_id(state.series.series_id, at)
    created = (
        None
        if existing is not None
        else _task_from_create(
            copy_task_create(occurrence_request(state.series, at)),
            task_id=task_id,
            parent_task=None,
            retry_started_at=now,
            supports_verified_work_contracts=True,
            event_budget=True,
        )
    )
    return state.model_copy(update={"next_task_id": task_id, "updated_at": now}), created


def _too_many_cancelled(state: TaskSeriesState) -> ValueError:
    return ValueError(
        f"Series {state.series.series_id!r} has more than {_CANCELLED_SKIP_MAX} ended "
        "occurrences ahead."
    )


def advance_series(
    state: TaskSeriesState,
    *,
    after: datetime | None,
    now: datetime,
    load: Callable[[str], Task | None],
) -> tuple[TaskSeriesState, Task | None]:
    """The state after queueing the next occurrence, and the task to insert (``None``: it exists)."""

    at = planned_occurrence(state.series, after=after, now=now)
    for _ in range(_CANCELLED_SKIP_MAX):
        existing = None if at is None else load(occurrence_task_id(state.series.series_id, at))
        if at is None or _taken(existing):
            return _advanced(state, at, existing, now)
        at = planned_occurrence(state.series, after=at, now=now)
    raise _too_many_cancelled(state)


async def advance_series_async(
    state: TaskSeriesState,
    *,
    after: datetime | None,
    now: datetime,
    load: Callable[[str], Awaitable[Task | None]],
) -> tuple[TaskSeriesState, Task | None]:
    """:func:`advance_series` for a store that reads tasks asynchronously."""

    at = planned_occurrence(state.series, after=after, now=now)
    for _ in range(_CANCELLED_SKIP_MAX):
        existing = (
            None if at is None else await load(occurrence_task_id(state.series.series_id, at))
        )
        if at is None or _taken(existing):
            return _advanced(state, at, existing, now)
        at = planned_occurrence(state.series, after=at, now=now)
    raise _too_many_cancelled(state)
