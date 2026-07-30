from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right, insort
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import islice
from typing import Any, ClassVar, Literal
from uuid import uuid4

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
    copy_durable_json_object,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.runtime.aggregates import EXACT_AGGREGATE, AggregateAccuracy, AggregateCount


class TaskStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskClaimLost(ValueError):
    """A worker no longer owns the active lease required for a task mutation."""


class TaskOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"


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
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
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

    @field_validator("input", "metadata", mode="before")
    @classmethod
    def copy_json_object(cls, value: dict[str, Any], info) -> dict[str, Any]:
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


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str | None = None
    type: str
    title: str | None = None
    description: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input", "metadata", mode="before")
    @classmethod
    def copy_json_object(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_durable_json_object(value, info.field_name)

    @field_validator("type")
    @classmethod
    def validate_nonblank_type(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "task_id",
        "title",
        "description",
        "session_id",
        "parent_task_id",
        "assigned_agent_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name in {"title", "description"}:
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)


class TaskQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: TaskStatus | None = None
    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
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
    accuracy: AggregateAccuracy

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
        return self


TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS = 50
TASK_TOPOLOGY_MAX_EXPANDED_PARENTS = 50
TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT = 25
TASK_TOPOLOGY_MAX_BRANCH_LIMIT = 100
TASK_TOPOLOGY_MAX_NODES = 500
TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES = 1024
TASK_TOPOLOGY_MAX_CURSOR_BYTES = 4096
TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES = 4096
TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH = 128
TASK_TOPOLOGY_MAX_VALIDATION_NODES = 4096
TaskTopologyTruncatedField = Literal[
    "type",
    "title",
    "assigned_agent_name",
    "status_reason",
]


class TaskTopologyCycle(ValueError):
    """Durable parent-task records contain a cycle reachable from the projection."""


class TaskTopologyInconsistent(ValueError):
    """Durable task records cannot form a truthful bounded topology projection."""


class TaskTopologyTraversalLimitExceeded(ValueError):
    """Task ancestry cannot be validated within the bounded topology contract."""


def _bounded_task_topology_text(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
    allow_controls: bool = False,
) -> str:
    validator = require_nonblank if allow_controls else require_clean_nonblank
    value = validator(value, field_name)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain portable Unicode text.") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds the task topology byte limit.")
    return value


def _bounded_task_topology_display(
    value: str | None,
    field_name: TaskTopologyTruncatedField,
    *,
    allow_controls: bool,
) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    try:
        return (
            _bounded_task_topology_text(
                value,
                field_name,
                max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
                allow_controls=allow_controls,
            ),
            False,
        )
    except ValueError:
        # Oversized display text is omitted rather than copied into the bounded
        # projection. The explicit marker keeps absence distinct from truncation.
        if len(value.encode("utf-8", errors="surrogatepass")) > (
            TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES
        ):
            return None, True
        raise


class TaskTopologyNode(BaseModel):
    """Payload-free bounded identity for one task topology node."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str | None
    title: str | None
    status: TaskStatus
    status_reason: str | None
    session_id: str | None
    parent_task_id: str | None
    assigned_agent_name: str | None
    created_at: datetime
    updated_at: datetime
    truncated_fields: tuple[TaskTopologyTruncatedField, ...] = ()

    @classmethod
    def from_task(cls, task: Task) -> TaskTopologyNode:
        if type(task) is not Task:
            raise TypeError("Task topology nodes require Task instances.")
        task_type, type_truncated = _bounded_task_topology_display(
            task.type,
            "type",
            allow_controls=False,
        )
        title, title_truncated = _bounded_task_topology_display(
            task.title,
            "title",
            allow_controls=True,
        )
        assigned_agent_name, agent_truncated = _bounded_task_topology_display(
            task.assigned_agent_name,
            "assigned_agent_name",
            allow_controls=False,
        )
        status_reason, reason_truncated = _bounded_task_topology_display(
            task.status_reason,
            "status_reason",
            allow_controls=True,
        )
        truncated_fields = tuple(
            field_name
            for field_name, truncated in (
                ("type", type_truncated),
                ("title", title_truncated),
                ("assigned_agent_name", agent_truncated),
                ("status_reason", reason_truncated),
            )
            if truncated
        )
        try:
            return cls(
                id=task.id,
                type=task_type,
                title=title,
                status=task.status,
                status_reason=status_reason,
                session_id=task.session_id,
                parent_task_id=task.parent_task_id,
                assigned_agent_name=assigned_agent_name,
                created_at=task.created_at,
                updated_at=task.updated_at,
                truncated_fields=truncated_fields,
            )
        except (TypeError, ValueError) as exc:
            raise TaskTopologyInconsistent(
                "A task record cannot be represented by the bounded topology contract."
            ) from exc

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("session_id", "parent_task_id")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("type", "assigned_agent_name")
    @classmethod
    def validate_optional_clean_display(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
        )

    @field_validator("title", "status_reason")
    @classmethod
    def validate_optional_display(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_task_topology_text(
            value,
            info.field_name,
            max_bytes=TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
            allow_controls=True,
        )

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("truncated_fields")
    @classmethod
    def validate_truncated_fields(
        cls,
        value: tuple[TaskTopologyTruncatedField, ...],
    ) -> tuple[TaskTopologyTruncatedField, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Task topology truncated_fields must not contain duplicates.")
        canonical = ("type", "title", "assigned_agent_name", "status_reason")
        if tuple(field for field in canonical if field in value) != value:
            raise ValueError("Task topology truncated_fields must use canonical order.")
        return value

    @model_validator(mode="after")
    def validate_display_omissions(self) -> TaskTopologyNode:
        truncated = set(self.truncated_fields)
        for field_name in truncated:
            if getattr(self, field_name) is not None:
                raise ValueError("Truncated task topology display fields must be omitted.")
        if self.type is None and "type" not in truncated:
            raise ValueError("Task topology type may be absent only when explicitly truncated.")
        return self


class TaskTopologyQuery(BaseModel):
    """Batched task links for explicitly expanded session and task branches."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    linked_session_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    session_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    expanded_parent_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_cursors: dict[str, str] = Field(
        default_factory=dict,
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    session_task_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    child_limit: StrictInt = Field(
        default=TASK_TOPOLOGY_DEFAULT_BRANCH_LIMIT,
        ge=1,
        le=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )

    @field_validator("linked_session_ids", "expanded_parent_ids", mode="before")
    @classmethod
    def copy_branch_ids(cls, value, info) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError(f"{info.field_name} must be a sequence of strings.")
        branch_limit = (
            TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS
            if info.field_name == "linked_session_ids"
            else TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
        )
        try:
            values = islice(iter(value), branch_limit + 1)
        except TypeError as exc:
            raise ValueError(f"{info.field_name} must be a sequence of strings.") from exc
        copied: list[str] = []
        for index, item in enumerate(values):
            if index == branch_limit:
                raise ValueError(f"{info.field_name} exceeds its task topology branch limit.")
            if type(item) is not str:
                raise ValueError(f"{info.field_name} must contain only strings.")
            item = _bounded_task_topology_text(
                item,
                f"{info.field_name}[{index}]",
                max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            if item in copied:
                raise ValueError(f"{info.field_name} must not contain duplicates.")
            copied.append(item)
        return tuple(copied)

    @field_validator("session_cursors", "child_cursors", mode="before")
    @classmethod
    def copy_cursors(cls, value, info) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise ValueError(f"{info.field_name} must be an object.")
        cursor_limit = (
            TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS
            if info.field_name == "session_cursors"
            else TASK_TOPOLOGY_MAX_EXPANDED_PARENTS
        )
        if len(value) > cursor_limit:
            raise ValueError(f"{info.field_name} exceeds its task topology branch limit.")
        copied: dict[str, str] = {}
        for raw_parent_id, raw_cursor in value.items():
            if type(raw_parent_id) is not str or type(raw_cursor) is not str:
                raise ValueError(f"{info.field_name} must map strings to strings.")
            parent_id = _bounded_task_topology_text(
                raw_parent_id,
                f"{info.field_name} key",
                max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
            )
            copied[parent_id] = _bounded_task_topology_text(
                raw_cursor,
                f"{info.field_name}[{parent_id!r}]",
                max_bytes=TASK_TOPOLOGY_MAX_CURSOR_BYTES,
            )
        return copied

    @model_validator(mode="after")
    def validate_cursor_authority(self) -> TaskTopologyQuery:
        if set(self.session_cursors).difference(self.linked_session_ids):
            raise ValueError("session_cursors keys must also appear in linked_session_ids.")
        if set(self.child_cursors).difference(self.expanded_parent_ids):
            raise ValueError("child_cursors keys must also appear in expanded_parent_ids.")
        return self


def _allocate_task_topology_branch_limits(
    query: TaskTopologyQuery,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reserve the shared node budget before stores hydrate branch candidates.

    Every requested branch receives at least one return slot. Earlier branches
    receive their requested limit while capacity permits; later branches retain
    a slot and therefore always have a truthful continuation boundary. Each
    store reads one additional sentinel row per branch to determine ``has_more``.
    """

    if type(query) is not TaskTopologyQuery:
        raise TypeError("Task topology branch allocation requires a TaskTopologyQuery.")
    requested_limits = (
        *(query.session_task_limit for _ in query.linked_session_ids),
        *(query.child_limit for _ in query.expanded_parent_ids),
    )
    if not requested_limits:
        return (), ()

    remaining = TASK_TOPOLOGY_MAX_NODES - len(query.expanded_parent_ids)
    allocated: list[int] = []
    for index, requested_limit in enumerate(requested_limits):
        remaining_branches = len(requested_limits) - index - 1
        branch_limit = min(requested_limit, remaining - remaining_branches)
        if branch_limit < 1:
            raise RuntimeError("Task topology node allocation cannot retain every branch.")
        allocated.append(branch_limit)
        remaining -= branch_limit

    session_count = len(query.linked_session_ids)
    return tuple(allocated[:session_count]), tuple(allocated[session_count:])


class TaskTopologySessionBranch(BaseModel):
    """Tasks attached to one explicitly expanded session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    tasks: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "session_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologySessionBranch:
        if any(task.session_id != self.session_id for task in self.tasks):
            raise ValueError("A session-task branch contains a contradictory session link.")
        _validate_task_topology_page(
            self.tasks,
            self.next_cursor,
            self.has_more,
            scope_id=self.session_id,
            scope_kind="session",
        )
        return self


class TaskTopologyChildBranch(BaseModel):
    """Direct task children of one explicitly expanded task."""

    model_config = ConfigDict(extra="forbid")

    parent_task_id: str
    children: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_BRANCH_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("parent_task_id")
    @classmethod
    def validate_parent_task_id(cls, value: str) -> str:
        return _bounded_task_topology_text(
            value,
            "parent_task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologyChildBranch:
        if any(task.parent_task_id != self.parent_task_id for task in self.children):
            raise ValueError("A child-task branch contains a contradictory parent link.")
        _validate_task_topology_page(
            self.children,
            self.next_cursor,
            self.has_more,
            scope_id=self.parent_task_id,
            scope_kind="parent_task",
        )
        return self


class TaskTopologyStoreResult(BaseModel):
    """Backend-neutral bounded task projection captured by one task-store snapshot."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    session_branches: tuple[TaskTopologySessionBranch, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS,
    )
    expanded_parents: tuple[TaskTopologyNode, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )
    child_branches: tuple[TaskTopologyChildBranch, ...] = Field(
        default=(),
        max_length=TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    )

    @field_validator("observed_at")
    @classmethod
    def normalize_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_shape(self) -> TaskTopologyStoreResult:
        session_ids = [branch.session_id for branch in self.session_branches]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Task topology session branches must not contain duplicates.")
        expanded_ids = [node.id for node in self.expanded_parents]
        if len(set(expanded_ids)) != len(expanded_ids):
            raise ValueError("Task topology expanded parents must not contain duplicates.")
        if len(self.child_branches) != len(self.expanded_parents):
            raise ValueError("Every expanded task parent requires exactly one child branch.")
        if [branch.parent_task_id for branch in self.child_branches] != expanded_ids:
            raise ValueError("Task child branches must preserve expanded-parent order.")

        nodes = (
            *self.expanded_parents,
            *(task for branch in self.session_branches for task in branch.tasks),
            *(task for branch in self.child_branches for task in branch.children),
        )
        nodes_by_id: dict[str, TaskTopologyNode] = {}
        for node in nodes:
            prior = nodes_by_id.setdefault(node.id, node)
            if prior != node:
                raise TaskTopologyInconsistent(
                    "Task topology contains contradictory representations of one task."
                )
        if len(nodes_by_id) > TASK_TOPOLOGY_MAX_NODES:
            raise ValueError(
                f"Task topology cannot retain more than {TASK_TOPOLOGY_MAX_NODES} nodes."
            )
        _reject_loaded_task_topology_cycles(nodes_by_id)
        return self

    def validate_for_query(self, query: TaskTopologyQuery) -> TaskTopologyStoreResult:
        """Verify that a custom store honored the exact requested branches and bounds."""

        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology result validation requires a TaskTopologyQuery.")
        if tuple(branch.session_id for branch in self.session_branches) != (
            query.linked_session_ids
        ):
            raise TaskTopologyInconsistent(
                "Task topology session branches do not match the requested sessions."
            )
        if tuple(node.id for node in self.expanded_parents) != query.expanded_parent_ids:
            raise TaskTopologyInconsistent(
                "Task topology parents do not match the requested expansions."
            )
        for branch in self.session_branches:
            if len(branch.tasks) > query.session_task_limit:
                raise TaskTopologyInconsistent(
                    "A task topology session branch exceeds its requested limit."
                )
            cursor = query.session_cursors.get(branch.session_id)
            if cursor is not None and branch.tasks:
                boundary = decode_task_topology_cursor(
                    cursor,
                    scope_kind="session",
                    scope_id=branch.session_id,
                )
                if (branch.tasks[0].created_at, branch.tasks[0].id) <= boundary:
                    raise TaskTopologyInconsistent(
                        "A task topology session branch did not advance past its cursor."
                    )
        for branch in self.child_branches:
            if len(branch.children) > query.child_limit:
                raise TaskTopologyInconsistent(
                    "A task topology child branch exceeds its requested limit."
                )
            cursor = query.child_cursors.get(branch.parent_task_id)
            if cursor is not None and branch.children:
                boundary = decode_task_topology_cursor(
                    cursor,
                    scope_kind="parent_task",
                    scope_id=branch.parent_task_id,
                )
                if (branch.children[0].created_at, branch.children[0].id) <= boundary:
                    raise TaskTopologyInconsistent(
                        "A task topology child branch did not advance past its cursor."
                    )
        return self


class TaskStore(ABC):
    """Persistent store for durable work items."""

    supports_task_topology: ClassVar[bool] = False

    @abstractmethod
    async def create_task(self, request: TaskCreate) -> Task:
        """Create a task."""

    @abstractmethod
    async def create_running_task(self, request: TaskCreate) -> Task:
        """Atomically create a running task already attached to its session.

        ``request.session_id`` is required. This avoids leaving an attached,
        unclaimable pending task if a process stops between separate create and
        start operations.
        """

    @abstractmethod
    async def load_task(self, task_id: str) -> Task | None:
        """Load a task by id."""

    @abstractmethod
    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        """List tasks for dashboards, queues, and orchestration."""

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        """Count current task states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable when they do not expose this control-plane read model.
        """
        raise NotImplementedError(
            "This TaskStore does not support operational aggregate snapshots."
        )

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        """Read bounded task/session and direct-child branches.

        Default raises ``NotImplementedError`` so existing out-of-tree stores
        remain instantiable while advertising the operation as unsupported.
        """
        raise NotImplementedError("This TaskStore does not support topology queries.")

    @abstractmethod
    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        """Mark a pending task as running, optionally attached to a session."""

    @abstractmethod
    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        worker_id: str,
    ) -> Task:
        """Attach a live worker-claimed task to a session and mark it running.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as completed.

        If ``worker_id`` is given, the update raises ``TaskClaimLost`` unless that
        worker still owns an active lease on the task, so a worker that lost its
        lease cannot clobber a task another worker has since reclaimed.
        """

    @abstractmethod
    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as failed.

        If ``worker_id`` is given, the update raises ``TaskClaimLost`` unless that
        worker still owns an active lease on the task.
        """

    @abstractmethod
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or running task as cancelled."""

    @abstractmethod
    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Pause a pending or unattached running task until app code resumes it."""

    @abstractmethod
    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as blocked on an external dependency."""

    @abstractmethod
    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or unattached running task as waiting for human/operator input."""

    @abstractmethod
    async def resume_task(self, task_id: str) -> Task:
        """Return a paused, blocked, or attention-needed task to the pending queue."""

    @abstractmethod
    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        """Atomically claim the next pending task matching ``query``."""

    @abstractmethod
    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        """Extend a worker-owned active lease.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def release_task(self, task_id: str, worker_id: str) -> Task:
        """Release a claimed task back to pending and clear worker ownership.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        """Release worker ownership while preserving a running task's session link.

        Raise ``TaskClaimLost`` if ``worker_id`` no longer owns a live claim.
        """

    @abstractmethod
    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        """Return expired claimed task leases to pending."""


class InMemoryTaskStore(TaskStore):
    """In-process task store for tests, local development, and examples."""

    supports_task_topology: ClassVar[bool] = True

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}
        self._task_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._task_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _task_from_create(request)
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            return task.model_copy(deep=True)

    async def create_running_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _running_task_from_create(request)
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._store_task(task)
            return task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return task.model_copy(deep=True)

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        async with self._lock:
            tasks = [task for task in self._tasks.values() if _task_matches(task, query)]
            tasks = _sort_tasks(tasks, query.order_by)
            page = tasks[query.offset : query.offset + query.limit]
            return [task.model_copy(deep=True) for task in page]

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        async with self._lock:
            session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(
                query
            )
            expanded_parents: list[TaskTopologyNode] = []
            for parent_id in query.expanded_parent_ids:
                parent = self._tasks.get(parent_id)
                if parent is None:
                    raise KeyError(f"Task not found: {parent_id}")
                expanded_parents.append(TaskTopologyNode.from_task(parent))

            session_candidates: list[list[TaskTopologyNode]] = []
            for session_id, branch_limit in zip(
                query.linked_session_ids,
                session_branch_limits,
                strict=True,
            ):
                session_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_session.get(session_id, ()),
                            cursor=query.session_cursors.get(session_id),
                            scope_kind="session",
                            scope_id=session_id,
                            limit=branch_limit,
                        )
                    ]
                )

            child_candidates: list[list[TaskTopologyNode]] = []
            for parent, branch_limit in zip(
                expanded_parents,
                child_branch_limits,
                strict=True,
            ):
                child_candidates.append(
                    [
                        TaskTopologyNode.from_task(task)
                        for task in self._task_topology_candidates(
                            self._task_keys_by_parent.get(parent.id, ()),
                            cursor=query.child_cursors.get(parent.id),
                            scope_kind="parent_task",
                            scope_id=parent.id,
                            limit=branch_limit,
                        )
                    ]
                )

            async def load_parent_links(task_ids: tuple[str, ...]) -> Mapping[str, str | None]:
                links: dict[str, str | None] = {}
                for task_id in task_ids:
                    task = self._tasks.get(task_id)
                    if task is None:
                        continue
                    links[task_id] = _bounded_optional_task_topology_parent_id(task.parent_task_id)
                return links

            await _validate_task_topology_ancestry(
                (
                    *expanded_parents,
                    *(task for branch in session_candidates for task in branch),
                    *(task for branch in child_candidates for task in branch),
                ),
                load_parent_links,
            )
            result = build_task_topology_result(
                observed_at=datetime.now(UTC),
                linked_session_ids=query.linked_session_ids,
                session_branch_candidates=session_candidates,
                session_branch_limits=session_branch_limits,
                expanded_parents=expanded_parents,
                child_branch_candidates=child_candidates,
                child_branch_limits=child_branch_limits,
                session_task_limit=query.session_task_limit,
                child_limit=query.child_limit,
            )
            return result

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        task_query = task_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = datetime.now(UTC)
            counts = {status: 0 for status in TaskStatus}
            total_count = 0
            for task in self._tasks.values():
                if _task_matches(task, task_query):
                    counts[task.status] += 1
                    total_count += 1
            return TaskOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=TaskStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            task = self._require_task(task_id)
            now = datetime.now(UTC)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": (session_id if session_id is not None else task.session_id),
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        worker_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_task(task_id)
            now = datetime.now(UTC)
            if not _can_attach_claimed_task(task, worker_id=worker_id, now=now):
                _raise_task_claim_attach_error(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": session_id,
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
            )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
            )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.PAUSED,
            reason=reason,
            payload=payload,
        )

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.BLOCKED,
            reason=reason,
            payload=payload,
        )

    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.NEEDS_ATTENTION,
            reason=reason,
            payload=payload,
        )

    async def resume_task(self, task_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_can_resume_task(task)
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "status_reason": None,
                    "status_payload": None,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        async with self._lock:
            candidates = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.PENDING
                and task.session_id is None
                and _task_matches_claim_filter(task, query)
            ]
            if not candidates:
                return None
            # Claiming is always FIFO by creation time, independent of the query's
            # display ordering, so the oldest pending task is dispatched first.
            task = _sort_tasks(candidates, TaskOrder.CREATED_AT_ASC)[0]
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.CLAIMED,
                    "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        extend_seconds = _validate_positive_int(extend_seconds, "extend_seconds")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "lease_expires_at": now + timedelta(seconds=extend_seconds),
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def release_task(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            if task.session_id is not None:
                raise ValueError(
                    f"Task {task.id} is already attached to session {task.session_id}."
                )
            if task.status is not TaskStatus.CLAIMED:
                raise ValueError(f"Task {task.id} is not claimed.")
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.PENDING,
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        async with self._lock:
            task = self._require_owned_leased_task(task_id, worker_id)
            if task.status is not TaskStatus.RUNNING:
                raise ValueError(f"Task {task.id} is not running.")
            if task.session_id is None:
                raise ValueError(f"Task {task.id} is not attached to a session.")
            now = datetime.now(UTC)
            _ensure_active_task_lease(task, worker_id, now=now)
            updated = task.model_copy(
                update={
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        async with self._lock:
            now = datetime.now(UTC)
            expired = [
                task
                for task in self._tasks.values()
                if task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and task.lease_expires_at is not None
                and task.lease_expires_at <= now
                and _task_matches_claim_filter(task, query)
            ]
            expired = _sort_tasks(expired, TaskOrder.UPDATED_AT_ASC)
            reclaimed: list[Task] = []
            for task in expired[:max_reclaims]:
                updated = task.model_copy(
                    update={
                        "status": TaskStatus.PENDING,
                        "worker_id": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                )
                self._store_task(updated)
                reclaimed.append(updated.model_copy(deep=True))
            return reclaimed

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _require_owned_leased_task(self, task_id: str, worker_id: str) -> Task:
        task = self._require_task(task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        return task

    def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
    ) -> Task:
        task = self._require_task(task_id)
        if worker_id is not None:
            if task.worker_id != worker_id:
                raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
            _ensure_active_task_lease(task, worker_id)
        _ensure_can_transition(task, status)
        now = datetime.now(UTC)
        updated = task.model_copy(
            update={
                "status": status,
                "status_reason": None,
                "status_payload": None,
                "result": deepcopy(result),
                "error": deepcopy(error),
                "worker_id": None,
                "lease_expires_at": None,
                "started_at": task.started_at or now,
                "completed_at": now,
                "updated_at": now,
            }
        )
        self._store_task(updated)
        return updated.model_copy(deep=True)

    async def _hold_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None,
        payload: dict[str, Any] | None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        reason = _copy_optional_status_reason(reason)
        payload = _copy_optional_status_payload(payload)
        async with self._lock:
            task = self._require_task(task_id)
            _ensure_can_hold_task(task, status)
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": status,
                    "status_reason": reason,
                    "status_payload": deepcopy(payload),
                    "worker_id": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self._store_task(updated)
            return updated.model_copy(deep=True)

    def _task_topology_candidates(
        self,
        keys: Sequence[tuple[datetime, str]],
        *,
        cursor: str | None,
        scope_kind: Literal["session", "parent_task"],
        scope_id: str,
        limit: int,
    ) -> list[Task]:
        start_index = 0
        if cursor is not None:
            cursor_created_at, cursor_id = decode_task_topology_cursor(
                cursor,
                scope_kind=scope_kind,
                scope_id=scope_id,
            )
            start_index = bisect_right(keys, (cursor_created_at, cursor_id))
        candidates: list[Task] = []
        for _, task_id in keys[start_index : start_index + limit + 1]:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index references a missing task."
                )
            if (scope_kind == "session" and task.session_id != scope_id) or (
                scope_kind == "parent_task" and task.parent_task_id != scope_id
            ):
                raise TaskTopologyInconsistent(
                    "The in-memory task topology index contains a contradictory link."
                )
            candidates.append(task)
        return candidates

    def _store_task(self, task: Task) -> None:
        prior = self._tasks.get(task.id)
        if prior is not None and (
            prior.created_at,
            prior.session_id,
            prior.parent_task_id,
        ) == (
            task.created_at,
            task.session_id,
            task.parent_task_id,
        ):
            # Lifecycle/status updates are the hot path. They do not change
            # either topology index, so avoid an O(n) list removal/reinsert.
            self._tasks[task.id] = task
            return
        if prior is not None:
            self._remove_task_index_entry(self._task_keys_by_session, prior.session_id, prior)
            self._remove_task_index_entry(self._task_keys_by_parent, prior.parent_task_id, prior)
        self._tasks[task.id] = task
        self._add_task_index_entry(self._task_keys_by_session, task.session_id, task)
        self._add_task_index_entry(self._task_keys_by_parent, task.parent_task_id, task)

    @staticmethod
    def _add_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        insort(index.setdefault(scope_id, []), (task.created_at, task.id))

    @staticmethod
    def _remove_task_index_entry(
        index: dict[str, list[tuple[datetime, str]]],
        scope_id: str | None,
        task: Task,
    ) -> None:
        if scope_id is None:
            return
        keys = index.get(scope_id)
        if keys is None:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        key = (task.created_at, task.id)
        position = bisect_left(keys, key)
        if position >= len(keys) or keys[position] != key:
            raise TaskTopologyInconsistent("The in-memory task topology index is incomplete.")
        keys.pop(position)
        if not keys:
            del index[scope_id]


def encode_task_topology_cursor(
    scope_kind: Literal["session", "parent_task"],
    scope_id: str,
    node: TaskTopologyNode,
) -> str:
    """Encode a scope-bound direct-link cursor."""

    if scope_kind not in {"session", "parent_task"}:
        raise ValueError("Invalid task topology cursor scope.")
    scope_id = _bounded_task_topology_text(
        scope_id,
        "scope_id",
        max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    if type(node) is not TaskTopologyNode:
        raise TypeError("Task topology cursors require TaskTopologyNode values.")
    raw = json.dumps(
        [
            scope_kind,
            scope_id,
            node.created_at.astimezone(UTC).isoformat(),
            node.id,
        ],
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    if len(encoded) > TASK_TOPOLOGY_MAX_CURSOR_BYTES:
        raise ValueError("Task topology cursor exceeds its byte limit.")
    return encoded


def decode_task_topology_cursor(
    cursor: str,
    *,
    scope_kind: Literal["session", "parent_task"],
    scope_id: str,
) -> tuple[datetime, str]:
    """Decode a task cursor and reject reuse against a different branch."""

    if scope_kind not in {"session", "parent_task"}:
        raise ValueError("Invalid task topology cursor scope.")
    scope_id = _bounded_task_topology_text(
        scope_id,
        "scope_id",
        max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    )
    try:
        cursor = _bounded_task_topology_text(
            cursor,
            "cursor",
            max_bytes=TASK_TOPOLOGY_MAX_CURSOR_BYTES,
        )
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical task topology cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid task topology cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 4
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
        or type(decoded[2]) is not str
        or type(decoded[3]) is not str
        or decoded[0] != scope_kind
        or decoded[1] != scope_id
        or not decoded[3]
    ):
        raise ValueError("Invalid task topology cursor.")
    try:
        task_id = _bounded_task_topology_text(
            decoded[3],
            "cursor task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )
        created_at = datetime.fromisoformat(decoded[2])
    except ValueError as exc:
        raise ValueError("Invalid task topology cursor.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Invalid task topology cursor.")
    return created_at.astimezone(UTC), task_id


def _validate_task_topology_page(
    tasks: tuple[TaskTopologyNode, ...],
    next_cursor: str | None,
    has_more: bool,
    *,
    scope_id: str,
    scope_kind: Literal["session", "parent_task"],
) -> None:
    task_ids = [task.id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("A task topology branch must not repeat a task.")
    if list(tasks) != sorted(tasks, key=lambda task: (task.created_at, task.id)):
        raise ValueError("Task topology branches must use stable creation ordering.")
    if has_more and next_cursor is None:
        raise ValueError("A task topology branch with more rows requires a cursor.")
    if not has_more and next_cursor is not None:
        raise ValueError("A complete task topology branch cannot expose a cursor.")
    if next_cursor is not None:
        cursor_created_at, cursor_id = decode_task_topology_cursor(
            next_cursor,
            scope_kind=scope_kind,
            scope_id=scope_id,
        )
        if not tasks or (cursor_created_at, cursor_id) != (
            tasks[-1].created_at,
            tasks[-1].id,
        ):
            raise ValueError(
                "A task topology continuation cursor must identify the last returned task."
            )


def build_task_topology_result(
    *,
    observed_at: datetime,
    linked_session_ids: Iterable[str],
    session_branch_candidates: Iterable[Iterable[TaskTopologyNode]],
    session_branch_limits: Iterable[int],
    expanded_parents: Iterable[TaskTopologyNode],
    child_branch_candidates: Iterable[Iterable[TaskTopologyNode]],
    child_branch_limits: Iterable[int],
    session_task_limit: int,
    child_limit: int,
) -> TaskTopologyStoreResult:
    """Apply the shared task-node ceiling without losing branch continuation."""

    for value, field_name in (
        (session_task_limit, "session_task_limit"),
        (child_limit, "child_limit"),
    ):
        if type(value) is not int or value < 1 or value > TASK_TOPOLOGY_MAX_BRANCH_LIMIT:
            raise ValueError(f"{field_name} is outside the task topology bounds.")

    session_ids = tuple(islice(linked_session_ids, TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS + 1))
    if len(session_ids) > TASK_TOPOLOGY_MAX_EXPANDED_SESSIONS:
        raise ValueError("Task topology exceeds the linked-session bound.")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Task topology linked sessions must not contain duplicates.")
    allocated_session_limits = _copy_task_topology_branch_limits(
        session_branch_limits,
        branch_count=len(session_ids),
        requested_limit=session_task_limit,
        field_name="session_branch_limits",
    )
    session_pages = tuple(
        tuple(islice(page, allocated_limit + 1))
        for page, allocated_limit in zip(
            islice(session_branch_candidates, len(session_ids) + 1),
            allocated_session_limits,
            strict=True,
        )
    )
    if len(session_pages) != len(session_ids):
        raise ValueError("Every linked session requires one task candidate page.")

    expanded_nodes = tuple(islice(expanded_parents, TASK_TOPOLOGY_MAX_EXPANDED_PARENTS + 1))
    if len(expanded_nodes) > TASK_TOPOLOGY_MAX_EXPANDED_PARENTS:
        raise ValueError("Task topology exceeds the expanded-parent bound.")
    allocated_child_limits = _copy_task_topology_branch_limits(
        child_branch_limits,
        branch_count=len(expanded_nodes),
        requested_limit=child_limit,
        field_name="child_branch_limits",
    )
    child_pages = tuple(
        tuple(islice(page, allocated_limit + 1))
        for page, allocated_limit in zip(
            islice(child_branch_candidates, len(expanded_nodes) + 1),
            allocated_child_limits,
            strict=True,
        )
    )
    if len(child_pages) != len(expanded_nodes):
        raise ValueError("Every expanded task parent requires one candidate page.")

    all_pages = (*session_pages, *child_pages)
    nonempty_after = [
        sum(bool(later) for later in all_pages[index + 1 :]) for index in range(len(all_pages))
    ]
    retained_ids = {node.id for node in expanded_nodes}

    session_branches: list[TaskTopologySessionBranch] = []
    page_index = 0
    for session_id, candidates, allocated_limit in zip(
        session_ids,
        session_pages,
        allocated_session_limits,
        strict=True,
    ):
        retained = _retain_task_topology_page(
            candidates,
            retained_ids=retained_ids,
            reserve_unique_slots=nonempty_after[page_index],
            limit=allocated_limit,
        )
        page_index += 1
        has_more = len(candidates) > len(retained)
        session_branches.append(
            TaskTopologySessionBranch(
                session_id=session_id,
                tasks=retained,
                next_cursor=(
                    encode_task_topology_cursor("session", session_id, retained[-1])
                    if has_more
                    else None
                ),
                has_more=has_more,
            )
        )

    child_branches: list[TaskTopologyChildBranch] = []
    for parent, candidates, allocated_limit in zip(
        expanded_nodes,
        child_pages,
        allocated_child_limits,
        strict=True,
    ):
        retained = _retain_task_topology_page(
            candidates,
            retained_ids=retained_ids,
            reserve_unique_slots=nonempty_after[page_index],
            limit=allocated_limit,
        )
        page_index += 1
        has_more = len(candidates) > len(retained)
        child_branches.append(
            TaskTopologyChildBranch(
                parent_task_id=parent.id,
                children=retained,
                next_cursor=(
                    encode_task_topology_cursor("parent_task", parent.id, retained[-1])
                    if has_more
                    else None
                ),
                has_more=has_more,
            )
        )

    loaded_nodes = (
        *expanded_nodes,
        *(task for branch in session_branches for task in branch.tasks),
        *(task for branch in child_branches for task in branch.children),
    )
    loaded_nodes_by_id: dict[str, TaskTopologyNode] = {}
    for node in loaded_nodes:
        prior = loaded_nodes_by_id.setdefault(node.id, node)
        if prior != node:
            raise TaskTopologyInconsistent(
                "Task topology contains contradictory representations of one task."
            )
    _reject_loaded_task_topology_cycles(loaded_nodes_by_id)

    return TaskTopologyStoreResult(
        observed_at=observed_at,
        session_branches=tuple(session_branches),
        expanded_parents=expanded_nodes,
        child_branches=tuple(child_branches),
    )


def _copy_task_topology_branch_limits(
    values: Iterable[int],
    *,
    branch_count: int,
    requested_limit: int,
    field_name: str,
) -> tuple[int, ...]:
    copied = tuple(islice(values, branch_count + 1))
    if len(copied) != branch_count:
        raise ValueError(f"{field_name} must provide exactly one limit per branch.")
    if any(type(value) is not int or value < 1 or value > requested_limit for value in copied):
        raise ValueError(f"{field_name} contains an invalid allocated branch limit.")
    return copied


def _retain_task_topology_page(
    candidates: tuple[TaskTopologyNode, ...],
    *,
    retained_ids: set[str],
    reserve_unique_slots: int,
    limit: int,
) -> tuple[TaskTopologyNode, ...]:
    available_unique = TASK_TOPOLOGY_MAX_NODES - len(retained_ids)
    unique_capacity = max(0, available_unique - reserve_unique_slots)
    retained: list[TaskTopologyNode] = []
    new_ids: set[str] = set()
    for candidate in candidates[:limit]:
        is_new = candidate.id not in retained_ids and candidate.id not in new_ids
        if is_new and len(new_ids) >= unique_capacity:
            break
        retained.append(candidate)
        if is_new:
            new_ids.add(candidate.id)
    if candidates and not retained:
        raise RuntimeError("Task topology node allocation could not retain a branch cursor.")
    retained_ids.update(new_ids)
    return tuple(retained)


def _reject_loaded_task_topology_cycles(
    nodes_by_id: Mapping[str, TaskTopologyNode],
) -> None:
    _reject_task_parent_link_cycles(
        {node_id: node.parent_task_id for node_id, node in nodes_by_id.items()}
    )


def _bounded_optional_task_topology_parent_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return _bounded_task_topology_text(
            value,
            "parent_task_id",
            max_bytes=TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task topology record contains an invalid durable parent identifier."
        ) from exc


async def _validate_task_topology_ancestry(
    seed_nodes: Iterable[TaskTopologyNode],
    load_parent_links: Callable[
        [tuple[str, ...]],
        Awaitable[Mapping[str, str | None]],
    ],
) -> None:
    """Validate complete parent chains for projected candidates under hard bounds."""

    parent_by_id: dict[str, str | None] = {}
    for node in seed_nodes:
        prior = parent_by_id.setdefault(node.id, node.parent_task_id)
        if prior != node.parent_task_id:
            raise TaskTopologyInconsistent(
                "Task topology contains contradictory parent links for one task."
            )

    frontier = {
        parent_id
        for parent_id in parent_by_id.values()
        if parent_id is not None and parent_id not in parent_by_id
    }
    depth = 0
    while frontier:
        if depth >= TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH:
            raise TaskTopologyTraversalLimitExceeded(
                "Task topology ancestry exceeds its depth limit."
            )
        task_ids = tuple(sorted(frontier))
        if len(parent_by_id) + len(task_ids) > TASK_TOPOLOGY_MAX_VALIDATION_NODES:
            raise TaskTopologyTraversalLimitExceeded(
                "Task topology ancestry exceeds its validation-node limit."
            )
        loaded = await load_parent_links(task_ids)
        if set(loaded) != set(task_ids):
            raise TaskTopologyInconsistent(
                "A task topology record references a missing durable parent."
            )
        for task_id in task_ids:
            parent_by_id[task_id] = _bounded_optional_task_topology_parent_id(loaded[task_id])
        frontier = {
            parent_id
            for parent_id in (parent_by_id[task_id] for task_id in task_ids)
            if parent_id is not None and parent_id not in parent_by_id
        }
        depth += 1

    _reject_task_parent_link_cycles(parent_by_id)


def _reject_task_parent_link_cycles(
    parent_by_id: Mapping[str, str | None],
) -> None:
    complete: set[str] = set()
    for start_id in parent_by_id:
        if start_id in complete:
            continue
        path: list[str] = []
        path_positions: dict[str, int] = {}
        current_id: str | None = start_id
        while current_id is not None and current_id in parent_by_id:
            if current_id in complete:
                break
            if current_id in path_positions:
                raise TaskTopologyCycle("Task topology contains a cycle among loaded task nodes.")
            path_positions[current_id] = len(path)
            path.append(current_id)
            current_id = parent_by_id[current_id]
        complete.update(path)


def copy_task(task: Task) -> Task:
    if type(task) is not Task:
        raise TypeError("Tasks must be Task instances.")
    return Task(
        id=task.id,
        type=task.type,
        title=task.title,
        description=task.description,
        status=task.status,
        session_id=task.session_id,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        status_reason=task.status_reason,
        status_payload=(
            None
            if task.status_payload is None
            else copy_durable_json_object(task.status_payload, "status_payload")
        ),
        input=copy_durable_json_object(task.input, "input"),
        result=(None if task.result is None else copy_durable_json_object(task.result, "result")),
        error=None if task.error is None else copy_durable_json_object(task.error, "error"),
        metadata=copy_durable_json_object(task.metadata, "metadata"),
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


def copy_task_create(request: TaskCreate) -> TaskCreate:
    if type(request) is not TaskCreate:
        raise TypeError("Task creation requires a TaskCreate instance.")
    return TaskCreate(
        task_id=request.task_id,
        type=request.type,
        title=request.title,
        description=request.description,
        session_id=request.session_id,
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_json_object(request.metadata, "metadata"),
    )


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


def _task_from_create(request: TaskCreate) -> Task:
    now = datetime.now(UTC)
    return Task(
        id=request.task_id if request.task_id is not None else str(uuid4()),
        type=request.type,
        title=request.title,
        description=request.description,
        status=TaskStatus.PENDING,
        session_id=request.session_id,
        parent_task_id=request.parent_task_id,
        assigned_agent_name=request.assigned_agent_name,
        input=copy_durable_json_object(request.input, "input"),
        metadata=copy_durable_json_object(request.metadata, "metadata"),
        created_at=now,
        updated_at=now,
    )


def _running_task_from_create(request: TaskCreate) -> Task:
    task = _task_from_create(request)
    if task.session_id is None:
        raise ValueError("TaskCreate.session_id is required to create a running task.")
    return task.model_copy(
        update={
            "status": TaskStatus.RUNNING,
            "started_at": task.created_at,
        }
    )


def _ensure_can_transition(task: Task, next_status: TaskStatus) -> None:
    if task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")
    if next_status == TaskStatus.RUNNING and task.status != TaskStatus.PENDING:
        raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")


def _ensure_can_hold_task(task: Task, next_status: TaskStatus) -> None:
    if next_status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} cannot be held as {next_status}.")
    _ensure_not_terminal(task)
    if task.status is TaskStatus.RUNNING and task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    if task.status not in {
        TaskStatus.PENDING,
        TaskStatus.CLAIMED,
        TaskStatus.RUNNING,
        *_HELD_TASK_STATUSES,
    }:
        raise ValueError(f"Task {task.id} cannot transition to {next_status} from {task.status}")


def _ensure_can_resume_task(task: Task) -> None:
    _ensure_not_terminal(task)
    if task.status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is not paused, blocked, or waiting for attention.")


def _ensure_not_terminal(task: Task) -> None:
    if task.status in _TERMINAL_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")


def _can_attach_claimed_task(
    task: Task,
    *,
    worker_id: str,
    now: datetime | None = None,
) -> bool:
    now = datetime.now(UTC) if now is None else now
    return (
        task.status is TaskStatus.CLAIMED
        and task.worker_id == worker_id
        and task.session_id is None
        and task.lease_expires_at is not None
        and task.lease_expires_at > now
    )


def _ensure_active_task_lease(task: Task, worker_id: str, *, now: datetime | None = None) -> None:
    now = datetime.now(UTC) if now is None else now
    if task.lease_expires_at is None:
        raise TaskClaimLost(f"Task {task.id} has no active lease.")
    if task.lease_expires_at <= now:
        raise TaskClaimLost(f"Task {task.id} lease for worker {worker_id} has expired.")


def _ensure_owned_active_task_lease(
    task: Task,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Require the supplied worker to own the task's current live lease."""

    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed or running.")
    if task.worker_id != worker_id:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    _ensure_active_task_lease(task, worker_id, now=now)


def _raise_task_claim_attach_error(
    task: Task,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> None:
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed by worker {worker_id}.")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.status is TaskStatus.RUNNING:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        raise ValueError(f"Task {task.id} is already running.")
    if task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    raise RuntimeError(f"Task {task.id} active claim could not be attached.")


def _task_matches(task: Task, query: TaskQuery) -> bool:
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


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value


def _copy_optional_status_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return require_nonblank(value, "reason")


def _copy_optional_status_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return copy_durable_json_object(value, "payload")


_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}

_HELD_TASK_STATUSES = {
    TaskStatus.PAUSED,
    TaskStatus.BLOCKED,
    TaskStatus.NEEDS_ATTENTION,
}
