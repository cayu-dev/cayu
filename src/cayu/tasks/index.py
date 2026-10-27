"""Bounded task pages and exact status counts from one native read snapshot."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

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
    counts: dict[TaskStatus, int] = field(default_factory=lambda: dict.fromkeys(TaskStatus, 0))
    total: int = 0
    has_more: bool = False

    def __post_init__(self) -> None:
        _reject_fenced_filter(self.query.filters)

    def visit(self, task: Task) -> None:
        if not self.query.include_internal and task.type.startswith("cayu."):
            return
        prerequisite = self.query.prerequisite_task_id
        if prerequisite is not None and prerequisite not in task.prerequisite_task_ids:
            return
        if not self.matches(task):
            return
        self.counts[task.status] += 1
        if self.query.filters.status is not None and task.status != self.query.filters.status:
            return
        if self.query.statuses and task.status not in self.query.statuses:
            return
        self.total += 1
        if self.query.after is None and self.total <= self.query.filters.offset:
            return
        key = task_index_key(task, self.query.filters.order_by)
        if self.query.after is not None:
            timestamp, identity = self.query.after
            ascending = self.query.filters.order_by in {
                TaskOrder.CREATED_AT_ASC,
                TaskOrder.UPDATED_AT_ASC,
            }
            after = key[0] > timestamp if ascending else key[0] < timestamp
            if not (after or (key[0] == timestamp and key[1] > identity)):
                return
        if len(self.items) < self.query.filters.limit:
            self.items.append(task.model_copy(deep=True))
        else:
            self.has_more = True

    def result(self, as_of: datetime) -> TaskIndexResult:
        next_position = (
            encode_task_index_cursor(task_index_key(self.items[-1], self.query.filters.order_by))
            if self.has_more and self.items
            else None
        )
        return TaskIndexResult(
            items=self.items,
            total_count=self.total,
            counts_by_status=TaskStatusCounts.model_validate(self.counts),
            next_position=next_position,
            as_of=as_of,
        )


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
