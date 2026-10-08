"""Task queries, bounded aggregates and shared task selection rules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

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


class TaskQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: TaskStatus | None = None
    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    has_work_contract: StrictBool | None = None
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
        session_id=query.session_id,
        parent_task_id=query.parent_task_id,
        assigned_agent_name=query.assigned_agent_name,
        has_work_contract=query.has_work_contract,
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
    if (
        query.has_work_contract is not None
        and (task.work_contract is not None) != query.has_work_contract
    ):
        return False
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
