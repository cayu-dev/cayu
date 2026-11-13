"""Bounded task pages and exact status counts from one native read snapshot."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import AggregateCount
from cayu.tasks.queries import TaskOrder, TaskQuery, TaskStatusCounts
from cayu.tasks.records import Task, TaskStatus


class TaskIndexQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    filters: TaskQuery = Field(default_factory=TaskQuery)
    include_internal: StrictBool = True
    statuses: tuple[TaskStatus, ...] = ()
    prerequisite_task_id: str | None = None
    after: tuple[datetime, str] | None = None

    @field_validator("filters")
    @classmethod
    def reject_listing_only_filters(cls, value: TaskQuery) -> TaskQuery:
        _reject_fenced_filter(value)
        return value


def _reject_fenced_filter(filters: TaskQuery) -> None:
    # The index counts from one snapshot without the store's ownership clock.
    if filters.fenced_after_worker_loss is not None:
        raise ValueError(
            "Task index queries do not support fenced_after_worker_loss; use list_tasks."
        )


class TaskIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Task]
    total_count: AggregateCount = Field(ge=0)
    counts_by_status: TaskStatusCounts
    next_position: str | None
    as_of: datetime


TaskIndexPredicate = Callable[[Task], bool]

#: The prefix that marks Cayu's own task types. Stores index it as ``is_internal``.
INTERNAL_TASK_TYPE_PREFIX = "cayu."

#: The task fields text search reads, in the order the public projection shows them.
TASK_SEARCH_FIELDS: tuple[str, ...] = (
    "id",
    "type",
    "title",
    "description",
    "status",
    "session_id",
    "parent_task_id",
    "assigned_agent_name",
    "worker_id",
    "status_reason",
)

#: Decides whether a task's stored search fields, in ``TASK_SEARCH_FIELDS`` order,
#: match its owner's text search. A native index calls it only for tasks whose
#: stored text could match ``filters.q``.
TaskSearchMatch = Callable[[Sequence[str | None]], bool]

#: The most tasks one ``load_tasks`` call reads.
MAX_TASK_BATCH = 100


def require_task_identities(task_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate a batch of task identities."""

    if isinstance(task_ids, str) or not isinstance(task_ids, Sequence):
        raise TypeError("task_ids must be a sequence of task identities.")
    if len(task_ids) > MAX_TASK_BATCH:
        raise ValueError(f"At most {MAX_TASK_BATCH} task identities can be read at once.")
    return tuple(require_clean_nonblank(task_id, "task_id") for task_id in task_ids)


@dataclass
class TaskIndexPage:
    """Counts every eligible task and keeps the identities of one page.

    The caller visits eligible tasks in query order, after every filter that
    applies before counting (all but ``status`` and ``statuses``).
    """

    query: TaskIndexQuery
    #: The keyset position keys compare with; ``query.after`` unless the caller
    #: visits keys in another representation that orders the same way.
    after: tuple[Any, str] | None = None
    keys: list[tuple[Any, str]] = field(default_factory=list)
    counts: dict[TaskStatus, int] = field(default_factory=lambda: dict.fromkeys(TaskStatus, 0))
    total: int = 0
    has_more: bool = False

    def __post_init__(self) -> None:
        if self.after is None:
            self.after = self.query.after

    def visit(self, status: TaskStatus, key: tuple[Any, str]) -> bool:
        """Count ``status`` and return whether the task belongs on the page."""

        self.counts[status] += 1
        filters = self.query.filters
        if filters.status is not None and status != filters.status:
            return False
        if self.query.statuses and status not in self.query.statuses:
            return False
        self.total += 1
        if self.after is None and self.total <= filters.offset:
            return False
        if self.after is not None:
            timestamp, identity = self.after
            ascending = filters.order_by in {TaskOrder.CREATED_AT_ASC, TaskOrder.UPDATED_AT_ASC}
            after = key[0] > timestamp if ascending else key[0] < timestamp
            if not (after or (key[0] == timestamp and key[1] > identity)):
                return False
        if len(self.keys) < filters.limit:
            self.keys.append(key)
            return True
        self.has_more = True
        return False

    def result(self, items: list[Task], as_of: datetime) -> TaskIndexResult:
        next_position = (
            encode_task_index_cursor(self.keys[-1]) if self.has_more and self.keys else None
        )
        return TaskIndexResult(
            items=items,
            total_count=self.total,
            counts_by_status=TaskStatusCounts.model_validate(self.counts),
            next_position=next_position,
            as_of=as_of,
        )


@dataclass
class TaskIndexAccumulator:
    """Retain only one page while visiting every eligible record exactly once.

    The native owner supplies records in deterministic query order from one
    snapshot, after resource access filtering. The public owner supplies the
    predicate for its redacted search representation; a private/raw text match
    is never an acceptable substitute for that predicate.
    """

    query: TaskIndexQuery
    matches: TaskIndexPredicate
    items: list[Task] = field(default_factory=list)
    page: TaskIndexPage = field(init=False)

    def __post_init__(self) -> None:
        _reject_fenced_filter(self.query.filters)
        self.page = TaskIndexPage(self.query)

    def visit(self, task: Task) -> None:
        if not self.query.include_internal and task.type.startswith(INTERNAL_TASK_TYPE_PREFIX):
            return
        prerequisite = self.query.prerequisite_task_id
        if prerequisite is not None and prerequisite not in task.prerequisite_task_ids:
            return
        if not self.matches(task):
            return
        if self.page.visit(task.status, task_index_key(task, self.query.filters.order_by)):
            self.items.append(task.model_copy(deep=True))

    def result(self, as_of: datetime) -> TaskIndexResult:
        return self.page.result(self.items, as_of)


def task_index_key(task: Task, order: TaskOrder) -> tuple[datetime, str]:
    timestamp = (
        task.created_at
        if order in {TaskOrder.CREATED_AT_ASC, TaskOrder.CREATED_AT_DESC}
        else task.updated_at
    )
    return timestamp, task.id


def encode_task_index_cursor(key: tuple[datetime, str]) -> str:
    return base64.urlsafe_b64encode(
        json.dumps([key[0].isoformat(), key[1]], separators=(",", ":")).encode()
    ).decode()


def decode_task_index_cursor(value: str) -> tuple[datetime, str]:
    if type(value) is not str or len(value) > 6144:
        raise ValueError("Invalid task cursor.")
    payload = json.loads(base64.b64decode(value, altchars=b"-_", validate=True))
    if (
        type(payload) is not list
        or len(payload) != 2
        or type(payload[0]) is not str
        or type(payload[1]) is not str
        or not payload[1]
    ):
        raise ValueError("Invalid task cursor.")
    timestamp = datetime.fromisoformat(payload[0])
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Invalid task cursor.")
    return timestamp, payload[1]


class TaskCountDifference(BaseModel):
    """One count where the stored task counts and a recount disagree."""

    model_config = ConfigDict(extra="forbid")

    internal: StrictBool
    status: str
    type: str
    assigned_agent_name: str | None
    stored: int
    recounted: int


class TaskCountsCheck(BaseModel):
    """The stored task counts by status, compared with a recount of every task."""

    model_config = ConfigDict(extra="forbid")

    by_status: dict[str, int]
    recounted_by_status: dict[str, int]
    differences: list[TaskCountDifference]
    matches: StrictBool


#: The most differences a counts check reports.
MAX_TASK_COUNT_DIFFERENCES = 100


def compare_task_counts(
    stored: Iterable[Sequence[Any]], recounted: Iterable[Sequence[Any]]
) -> TaskCountsCheck:
    """Compare ``(internal, status, type, agent or '', count)`` rows."""

    def totals(rows: Iterable[Sequence[Any]]) -> dict[tuple[bool, str, str, str], int]:
        result: dict[tuple[bool, str, str, str], int] = {}
        for internal, status, task_type, agent, count in rows:
            key = (bool(internal), str(status), str(task_type), str(agent))
            result[key] = result.get(key, 0) + int(count)
        return {key: count for key, count in result.items() if count}

    def by_status(counts: dict[tuple[bool, str, str, str], int]) -> dict[str, int]:
        result = dict.fromkeys((status.value for status in TaskStatus), 0)
        for (_, status, _, _), count in counts.items():
            result[status] = result.get(status, 0) + count
        return result

    kept, recount = totals(stored), totals(recounted)
    differences = [
        TaskCountDifference(
            internal=key[0],
            status=key[1],
            type=key[2],
            assigned_agent_name=key[3] or None,
            stored=kept.get(key, 0),
            recounted=recount.get(key, 0),
        )
        for key in sorted(set(kept) | set(recount))
        if kept.get(key, 0) != recount.get(key, 0)
    ]
    return TaskCountsCheck(
        by_status=by_status(kept),
        recounted_by_status=by_status(recount),
        differences=differences[:MAX_TASK_COUNT_DIFFERENCES],
        matches=not differences,
    )


async def require_unbounded_task_counts() -> None:
    """A counts check reads every task, so it is only for callers who can."""

    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.resource_access import current_data_bounds

    if await current_data_bounds("tasks") is not None:
        raise ResourceAccessDenied()
