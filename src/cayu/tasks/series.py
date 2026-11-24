"""Recurring task series: a cron or interval schedule that queues one task per occurrence.

A series is a definition the application keeps (for example per user, beside its
own settings): an id, when it recurs and the task to queue each time. Each part
works on its own:

- :class:`TaskRecurrence` says when a series recurs: a five-field ``cron``
  expression in an IANA ``timezone``, or ``every_seconds``, from ``start_at``.
- :func:`next_occurrence` and :func:`last_occurrence` compute occurrence times.
- :func:`publish_next_occurrence` queues the series' next occurrence as a managed
  scheduled task whose id comes from the series and the occurrence time, so
  publishing the same occurrence twice, also from racing callers or after a crash,
  makes one task.
- :func:`series_task_handler` wraps a task handler: before it runs an occurrence it
  publishes the next one, so the series keeps going whether the handler completes,
  fails or its worker is lost.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from cayu._validation import require_clean_nonblank
from cayu.tasks.creation import TaskCreate, copy_task_create
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.scheduling import TaskMisfirePolicy, TaskSchedulePolicy

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.tasks.store import TaskStore
    from cayu.tasks.worker import TaskHandler

#: The task metadata key that records which series and occurrence a task is.
SERIES_METADATA_KEY = "cayu_series"
#: The top-level metadata key every occurrence carries, with its series id as the value,
#: so ``cayu task list --metadata cayu_series_id=ID`` (and ``GET /api/tasks?metadata=``)
#: finds a series' occurrences. Metadata, not a label: a task's labels are the labels it
#: runs as, which decide what it may read and who may see it.
SERIES_ID_METADATA_KEY = "cayu_series_id"
# How far a cron search looks before deciding an expression never matches again.
_CRON_SEARCH_DAYS = 366 * 8
# How many cancelled future occurrences publication steps over before giving up.
_CANCELLED_SKIP_MAX = 100
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_DAYS = ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")


def _cron_field(text: str, low: int, high: int, names: tuple[str, ...] = ()) -> frozenset[int]:
    def number(value: str) -> int:
        upper = value.upper()
        if upper in names:
            return names.index(upper) + (1 if low == 1 else 0)
        if not value.isdigit():
            raise ValueError(f"{value!r} is not a number in a cron field.")
        return int(value)

    values: set[int] = set()
    for part in text.split(","):
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text.isdigit() else None
        if step_text and (step is None or step < 1):
            raise ValueError(f"{part!r} has an invalid step.")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            first, _, last = base.partition("-")
            start, end = number(first), number(last)
        else:
            start = number(base)
            end = high if step is not None else start
        if not low <= start <= end <= high:
            raise ValueError(f"{part!r} is outside {low}-{high}.")
        values.update(range(start, end + 1, step or 1))
    return frozenset(values)


class _Cron:
    def __init__(self, expression: str) -> None:
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError(
                "cron needs five fields: minute hour day-of-month month day-of-week, "
                "such as '0 9 * * MON'."
            )
        self.minutes = sorted(_cron_field(fields[0], 0, 59))
        self.hours = sorted(_cron_field(fields[1], 0, 23))
        self.days = _cron_field(fields[2], 1, 31)
        self.months = _cron_field(fields[3], 1, 12, _MONTHS)
        # 0 and 7 are both Sunday.
        self.weekdays = frozenset(day % 7 for day in _cron_field(fields[4], 0, 7, _DAYS))
        self.any_day = fields[2] == "*"
        self.any_weekday = fields[4] == "*"

    def matches_date(self, day: date) -> bool:
        if day.month not in self.months:
            return False
        in_month = day.day in self.days
        in_week = (day.isoweekday() % 7) in self.weekdays
        # As in cron: when both day fields are restricted, either may match.
        if not self.any_day and not self.any_weekday:
            return in_month or in_week
        return in_month and in_week

    def times(self, day: date, zone: ZoneInfo, *, reverse: bool) -> Iterator[datetime]:
        hours = reversed(self.hours) if reverse else self.hours
        for hour in hours:
            for minute in reversed(self.minutes) if reverse else self.minutes:
                local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
                moment = local.astimezone(UTC)
                # A local time a daylight-saving change skips doesn't occur.
                if moment.astimezone(zone).replace(tzinfo=None) == local.replace(tzinfo=None):
                    yield moment


class TaskRecurrence(BaseModel):
    """When a series recurs: ``cron`` in ``timezone``, or every ``every_seconds``.

    ``cron`` takes the five standard fields (minute, hour, day of month, month, day of
    week) with ``*``, lists, ranges, steps and three-letter names, such as
    ``"0 9 * * MON"`` (Mondays at 9:00) or ``"0 6 1 * *"`` (the 1st of each month at
    6:00). Times are read in ``timezone``; a local time that a daylight-saving change
    skips doesn't occur that day, and one it repeats occurs once. ``every_seconds``
    recurs at ``start_at`` and every interval after it. No occurrence is before
    ``start_at``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    cron: str | None = None
    every_seconds: StrictInt | None = Field(default=None, ge=60)
    timezone: str = "UTC"
    start_at: datetime

    @field_validator("start_at")
    @classmethod
    def normalize_start(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "start_at")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"{value!r} is not an IANA time zone name.") from exc
        return value

    @model_validator(mode="after")
    def validate_one_rule(self) -> TaskRecurrence:
        if (self.cron is None) == (self.every_seconds is None):
            raise ValueError("Set exactly one of cron and every_seconds.")
        if self.cron is not None:
            _Cron(self.cron)
        return self


def next_occurrence(recurrence: TaskRecurrence, after: datetime) -> datetime | None:
    """The first occurrence strictly after ``after``, or ``None`` if there is none."""

    after = normalize_utc_datetime(after, "after")
    start = recurrence.start_at
    if after < start:
        return start if recurrence.cron is None else _cron_search(recurrence, start, False, True)
    if recurrence.every_seconds is not None:
        steps = int((after - start).total_seconds() // recurrence.every_seconds) + 1
        return start + timedelta(seconds=steps * recurrence.every_seconds)
    return _cron_search(recurrence, after, False, False)


def last_occurrence(recurrence: TaskRecurrence, at_or_before: datetime) -> datetime | None:
    """The latest occurrence at or before ``at_or_before``, or ``None`` before the first."""

    at_or_before = normalize_utc_datetime(at_or_before, "at_or_before")
    start = recurrence.start_at
    if at_or_before < start:
        return None
    if recurrence.every_seconds is not None:
        steps = int((at_or_before - start).total_seconds() // recurrence.every_seconds)
        return start + timedelta(seconds=steps * recurrence.every_seconds)
    found = _cron_search(recurrence, at_or_before, True, True)
    return found if found is not None and found >= start else None


def _cron_search(
    recurrence: TaskRecurrence, edge: datetime, reverse: bool, inclusive: bool
) -> datetime | None:
    assert recurrence.cron is not None
    cron = _Cron(recurrence.cron)
    zone = ZoneInfo(recurrence.timezone)
    day = edge.astimezone(zone).date()
    step = timedelta(days=-1 if reverse else 1)
    for _ in range(_CRON_SEARCH_DAYS):
        if cron.matches_date(day):
            for moment in cron.times(day, zone, reverse=reverse):
                if (inclusive and moment == edge) or (moment < edge if reverse else moment > edge):
                    return moment
        day += step
    return None


class TaskSeries(BaseModel):
    """A recurring series: its id, when it recurs and the task each occurrence queues.

    ``task`` is the template, without ``task_id``, ``available_at``,
    ``schedule_policy``, ``retry_policy`` or a retry key; each occurrence copies it with its own id,
    time and policy, and records the series in ``metadata["cayu_series"]``. Labels,
    input and ``restartable`` carry over. ``misfire_policy`` says what to do with
    occurrences missed while nothing published or ran them: ``SKIP`` (the default)
    goes on with the next future occurrence, ``FIRE_ONCE`` runs the latest missed one
    once and then goes on. ``misfire_grace_seconds`` is how late an occurrence may
    start and still count as on time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    series_id: str = Field(max_length=200)
    recurrence: TaskRecurrence
    task: TaskCreate
    misfire_policy: TaskMisfirePolicy = TaskMisfirePolicy.SKIP
    misfire_grace_seconds: StrictInt = Field(default=60, ge=0, le=86_400)

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, value: str) -> str:
        value = require_clean_nonblank(value, "series_id")
        if "@" in value:
            raise ValueError("series_id can't contain '@'.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def validate_template(cls, value: object) -> TaskCreate:
        if isinstance(value, dict):
            value = TaskCreate.model_validate(value)
        if type(value) is not TaskCreate:
            raise TypeError("task must be a TaskCreate.")
        template = copy_task_create(value)
        # A retry series would need retry settlement where the handler wrapper completes
        # a paused or skipped occurrence, and its later attempts would look late.
        for name in (
            "task_id",
            "available_at",
            "schedule_policy",
            "retry_key",
            "event_id",
            "retry_policy",
        ):
            if getattr(template, name) is not None:
                raise ValueError(f"A series task template can't set {name}.")
        if SERIES_METADATA_KEY in template.metadata:
            raise ValueError(f"metadata[{SERIES_METADATA_KEY!r}] is set by the series.")
        return template

    @model_validator(mode="after")
    def validate_occurrences(self) -> TaskSeries:
        # Every occurrence is the template plus a schedule; refuse a template that
        # can't take one (for example one a managed schedule doesn't allow) up front.
        copy_task_create(occurrence_request(self, self.recurrence.start_at))
        return self


class TaskSeriesState(BaseModel):
    """A series as a task store keeps it (``TaskStore.save_task_series``): its definition, whether it is paused, its next task.

    ``next_task_id`` is the occurrence the store queued last, or ``None`` when the
    series has no further occurrence. ``revision`` goes up with every change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    series: TaskSeries
    paused: StrictBool = False
    revision: StrictInt = Field(ge=1)
    next_task_id: str | None = None
    updated_at: datetime


def occurrence_task_id(series_id: str, occurrence_at: datetime) -> str:
    """The task id of ``series_id``'s occurrence at ``occurrence_at``."""

    moment = normalize_utc_datetime(occurrence_at, "occurrence_at")
    return f"{series_id}@{moment:%Y%m%dT%H%M%SZ}"


def task_occurrence(task: Task) -> tuple[str, datetime] | None:
    """The series id and occurrence time ``task`` was queued for, or ``None``."""

    recorded = task.metadata.get(SERIES_METADATA_KEY)
    if type(recorded) is not dict:
        return None
    series_id, at = recorded.get("series_id"), recorded.get("occurrence_at")
    if type(series_id) is not str or type(at) is not str:
        return None
    return series_id, datetime.fromisoformat(at)


def planned_occurrence(
    series: TaskSeries, *, after: datetime | None, now: datetime
) -> datetime | None:
    """The occurrence to queue after ``after`` (``None``: from ``start_at``), by the misfire rule."""

    recurrence = series.recurrence
    at = next_occurrence(
        recurrence, recurrence.start_at - timedelta(microseconds=1) if after is None else after
    )
    if at is not None and (now - at).total_seconds() > series.misfire_grace_seconds:
        at = (
            next_occurrence(recurrence, now)
            if series.misfire_policy is TaskMisfirePolicy.SKIP
            else last_occurrence(recurrence, now)
        )
    return at


def occurrence_request(series: TaskSeries, at: datetime) -> TaskCreate:
    """The task creation request of ``series``' occurrence at ``at``."""

    return series.task.model_copy(
        update={
            "task_id": occurrence_task_id(series.series_id, at),
            "available_at": at,
            # The store always admits it, however late: series_task_handler applies
            # the series' misfire policy, so a skipped occurrence still publishes
            # the next one.
            "schedule_policy": TaskSchedulePolicy(
                misfire_policy=TaskMisfirePolicy.FIRE_ONCE,
                misfire_grace_seconds=series.misfire_grace_seconds,
            ),
            "metadata": {
                **series.task.metadata,
                SERIES_METADATA_KEY: {
                    "series_id": series.series_id,
                    "occurrence_at": at.isoformat(),
                },
                SERIES_ID_METADATA_KEY: series.series_id,
            },
        }
    )


async def publish_next_occurrence(
    task_store: TaskStore,
    series: TaskSeries,
    *,
    after: datetime | None = None,
    now: datetime | None = None,
) -> Task | None:
    """Queue the series' first occurrence after ``after``; return its task.

    ``after`` is usually the occurrence that is running; ``None`` starts from the
    series' ``start_at`` (that occurrence included). An occurrence already past its
    grace at ``now`` (the current time by default) is missed: with ``SKIP`` the next
    occurrence after ``now`` is queued instead, with ``FIRE_ONCE`` the latest missed
    one. The task's id comes from the series and the occurrence time, so a repeated or
    racing call for the same occurrence returns the task already queued, whatever
    state it is in now. Returns ``None`` when the series has no further occurrence.
    """

    series = TaskSeries.model_validate(series.model_dump())
    current = datetime.now(UTC) if now is None else normalize_utc_datetime(now, "now")
    recurrence = series.recurrence
    at = planned_occurrence(series, after=after, now=current)
    for _ in range(_CANCELLED_SKIP_MAX):
        if at is None:
            return None
        task = await _publish_occurrence(task_store, series, at)
        # A future occurrence someone cancelled (for example to pause) doesn't stop the
        # series: publishing moves on to the one after it.
        if task.status is not TaskStatus.CANCELLED or at <= current:
            return task
        at = next_occurrence(recurrence, at)
    raise ValueError(
        f"Series {series.series_id!r} has more than {_CANCELLED_SKIP_MAX} cancelled "
        "occurrences ahead; publish after the last of them."
    )


async def _publish_occurrence(task_store: TaskStore, series: TaskSeries, at: datetime) -> Task:
    task_id = occurrence_task_id(series.series_id, at)
    request = occurrence_request(series, at)
    existing = await task_store.load_task(task_id)
    if existing is not None:
        return existing
    try:
        return await task_store.create_task(request)
    except ValueError:
        # Another caller queued it first, or the template changed since: the
        # occurrence already has its task either way.
        existing = await task_store.load_task(task_id)
        if existing is None or task_occurrence(existing) != (series.series_id, at):
            raise
        return existing


SeriesLookup = Callable[[Task], TaskSeries | None | Awaitable[TaskSeries | None]]


def series_task_handler(
    task_store: TaskStore, handler: TaskHandler, *, series_for: SeriesLookup | None = None
) -> TaskHandler:
    """Wrap ``handler`` so it runs only the occurrences of active series, on time.

    Without ``series_for``, the series are the ones stored with
    ``TaskStore.save_task_series``, and the store queues each next occurrence when the
    current one ends. With ``series_for(task)``, which returns the task's series as the
    application keeps it now (sync or async), the wrapper publishes the next
    occurrence before it runs the current one, so a failure, a lost worker or a restart
    never stops the series. Either way, an occurrence of a paused or deleted series
    (``series_for`` returns ``None``) is completed with ``{"series_paused": true}``
    without running ``handler``, and one that starts more than
    ``misfire_grace_seconds`` late in a ``SKIP`` series with ``{"series_skipped":
    true}``. Tasks that aren't series occurrences go to ``handler`` unchanged.
    """

    from cayu.tasks.worker import complete_managed_task

    async def run(app: CayuApp, task: Task, worker_id: str) -> Any:
        occurrence = task_occurrence(task)
        if occurrence is None:
            return await handler(app, task, worker_id)
        if series_for is None:
            state = await task_store.load_task_series(occurrence[0])
            found = None if state is None or state.paused else state.series
        else:
            found = series_for(task)
            if inspect.isawaitable(found):
                found = await found
        if found is None:
            await complete_managed_task(task_store, task, worker_id, {"series_paused": True})
            return None
        if not isinstance(found, TaskSeries):
            raise TypeError("series_for must return a TaskSeries or None.")
        series = found
        now = datetime.now(UTC)
        if series_for is not None:
            await publish_next_occurrence(task_store, series, after=occurrence[1], now=now)
        late = (now - occurrence[1]).total_seconds() > series.misfire_grace_seconds
        # Only an occurrence's first run is skipped: a restart of it began on time.
        if late and series.misfire_policy is TaskMisfirePolicy.SKIP and task.restart_count == 0:
            await complete_managed_task(task_store, task, worker_id, {"series_skipped": True})
            return None
        return await handler(app, task, worker_id)

    return run


__all__ = [
    "SERIES_ID_METADATA_KEY",
    "SERIES_METADATA_KEY",
    "TaskRecurrence",
    "TaskSeries",
    "TaskSeriesState",
    "last_occurrence",
    "next_occurrence",
    "occurrence_task_id",
    "publish_next_occurrence",
    "series_task_handler",
    "task_occurrence",
]
