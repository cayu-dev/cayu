"""Bounded task topology contracts, cursors and result construction."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import UTC, datetime
from itertools import islice
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

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.tasks.records import Task, TaskStatus

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
