"""Task queries, bounded aggregates and shared task selection rules."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import AggregateAccuracy, AggregateCount
from cayu.tasks.contracts import validate_work_completion_linked_id
from cayu.tasks.records import (
    Task,
    TaskStatus,
)


class TaskOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"


# The most task types one query may name.
MAX_TASK_QUERY_TYPES = 50


class TaskQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: TaskStatus | None = None
    type: str | None = None
    # Several task types: a task of any of them matches, and a worker claims any of them
    # in the store's usual claim order. Give ``type`` or ``types``, not both.
    types: tuple[str, ...] | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    has_work_contract: StrictBool | None = None
    # True lists only tasks fenced after their worker's lease expired (see
    # ``is_fenced_after_worker_loss``); False leaves them out. Listing only.
    fenced_after_worker_loss: StrictBool | None = None
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

    @field_validator("types")
    @classmethod
    def validate_types(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not 1 <= len(value) <= MAX_TASK_QUERY_TYPES:
            raise ValueError(f"types must name 1 to {MAX_TASK_QUERY_TYPES} task types.")
        types = tuple(require_clean_nonblank(item, "types") for item in value)
        if len(set(types)) != len(types):
            raise ValueError("types must not repeat a task type.")
        return types

    @model_validator(mode="after")
    def validate_one_type_filter(self) -> TaskQuery:
        if self.type is not None and self.types is not None:
            raise ValueError("Give type or types, not both.")
        return self


def task_query_types(query: TaskQuery) -> tuple[str, ...] | None:
    """The task types ``query`` matches, or ``None`` when it matches every type."""

    if query.types is not None:
        return query.types
    return None if query.type is None else (query.type,)


# ``by`` for ``TaskStore.latest_tasks``: a field of the task's input, such as
# ``input.repository`` or ``input.target.repository``.
# A segment names an object key; one PostgreSQL could read as an array index (such as
# 0 or -1) would read an array element there.
_LATEST_BY_SEGMENT = re.compile(r"(?=[A-Za-z0-9_-]{1,64}$)[0-9-]*[A-Za-z_][A-Za-z0-9_-]*")
_MAX_LATEST_BY_DEPTH = 8


def latest_tasks_key_path(by: str) -> tuple[str, ...]:
    """The input path ``by`` names (``"input.repository"`` is ``("repository",)``)."""

    if not isinstance(by, str):
        raise TypeError("by must be a string such as 'input.repository'.")
    root, _, rest = by.partition(".")
    segments = tuple(rest.split(".")) if rest else ()
    if (
        root != "input"
        or not 1 <= len(segments) <= _MAX_LATEST_BY_DEPTH
        or not all(_LATEST_BY_SEGMENT.fullmatch(segment) for segment in segments)
    ):
        raise ValueError(
            "by must name a field of the task's input, such as 'input.repository': "
            f"up to {_MAX_LATEST_BY_DEPTH} names of letters, digits, '_' and '-', "
            "each with at least one letter or '_'."
        )
    return segments


def task_input_key(task_input: object, path: tuple[str, ...]) -> str | None:
    """The string at ``path`` in a task's input, or ``None`` when it isn't a string."""

    value: Any = task_input
    for segment in path:
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = cast("Mapping[str, Any]", value)[segment]
    return value if isinstance(value, str) else None


def select_latest_tasks(
    tasks: Iterable[Task], query: TaskQuery, path: tuple[str, ...] | None
) -> list[Task]:
    """The finished tasks newest first (by completion), one per input key with ``path``,
    and the page ``query.limit`` and ``query.offset`` select.

    Stores that can't do this natively use it on tasks already matched to ``query``.
    """

    finished = sorted(
        sorted((task for task in tasks if task.completed_at is not None), key=lambda t: t.id),
        key=lambda task: task.completed_at or datetime.min,
        reverse=True,
    )
    if path is not None:
        newest: dict[str, Task] = {}
        for task in finished:
            key = task_input_key(task.input, path)
            if key is not None and key not in newest:
                newest[key] = task
        finished = list(newest.values())
    return finished[query.offset : query.offset + query.limit]


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
    waiting_dependencies: AggregateCount = Field(default=0, ge=0)
    waiting_group: AggregateCount = Field(default=0, ge=0)
    dependency_skipped: AggregateCount = Field(default=0, ge=0)
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


def copy_task_query(query: TaskQuery | None) -> TaskQuery:
    if query is None:
        return TaskQuery()
    if type(query) is not TaskQuery:
        raise TypeError("Task queries must be TaskQuery instances.")
    return TaskQuery(
        q=query.q,
        status=query.status,
        type=query.type,
        types=query.types,
        session_id=query.session_id,
        parent_task_id=query.parent_task_id,
        assigned_agent_name=query.assigned_agent_name,
        has_work_contract=query.has_work_contract,
        fenced_after_worker_loss=query.fenced_after_worker_loss,
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


def _work_attempt_discovery_query(
    task_filter: TaskAggregateFilter | None, *, limit: int, after: str | None
) -> tuple[TaskQuery, str | None]:
    copied = copy_task_aggregate_filter(task_filter)
    query = TaskQuery(
        type=copied.type,
        session_id=copied.session_id,
        parent_task_id=copied.parent_task_id,
        assigned_agent_name=copied.assigned_agent_name,
        has_work_contract=True,
        limit=limit,
    )
    return query, None if after is None else validate_work_completion_linked_id(after, "after")


def _task_matches(task: Task, query: TaskQuery) -> bool:
    if (
        query.has_work_contract is not None
        and (task.work_contract is not None) != query.has_work_contract
    ):
        return False
    if query.q is not None and not _task_matches_search(task, query.q):
        return False
    if query.status is not None and task.status != query.status:
        return False
    types = task_query_types(query)
    if types is not None and task.type not in types:
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
    if (
        query.has_work_contract is not None
        and (task.work_contract is not None) != query.has_work_contract
    ):
        return False
    types = task_query_types(query)
    if types is not None and task.type not in types:
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
    if query.fenced_after_worker_loss is not None:
        raise ValueError("Task claim queries do not support fenced_after_worker_loss.")
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
