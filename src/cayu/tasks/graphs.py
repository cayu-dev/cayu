"""Bounded, self-contained task graphs; dependencies are exact task identities.

These contracts describe task eligibility, not task groups or an execution
engine. Graph events belong to the task store and do not require a session.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.tasks.base import TaskCreate, TaskInvocationSnapshot, TaskStatus, copy_task_create

TASK_GRAPH_MAX_NODES = 128
TASK_GRAPH_MAX_EDGES = 1024
TASK_GRAPH_MAX_BYTES = 1024 * 1024
TASK_GRAPH_ID_MAX_BYTES = 256


class TaskGraphConflict(ValueError):
    """A graph operation conflicts with retained identity or lifecycle authority."""


class TaskGraphUnavailable(ValueError):
    """Complete authoritative graph evidence is unavailable."""


def graph_identifier(value: str) -> str:
    value = require_durable_clean_nonblank(value, "graph identity")
    if len(value.encode("utf-8")) > TASK_GRAPH_ID_MAX_BYTES:
        raise ValueError("Graph identity exceeds its byte limit.")
    return value


GraphIdentifier = Annotated[str, AfterValidator(graph_identifier)]


def _canonical_identifiers(values: tuple[str, ...]) -> tuple[str, ...]:
    if values != tuple(sorted(set(values))):
        raise ValueError("Graph identities must be unique and canonically ordered.")
    return values


GraphIdentifiers = Annotated[
    tuple[GraphIdentifier, ...],
    Field(max_length=TASK_GRAPH_MAX_NODES),
    AfterValidator(_canonical_identifiers),
]


class TaskGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task: TaskCreate
    prerequisite_task_ids: tuple[str, ...] = Field(default=(), max_length=TASK_GRAPH_MAX_NODES)

    @field_validator("task", mode="before")
    @classmethod
    def copy_task_request(cls, value: object) -> TaskCreate:
        if isinstance(value, dict):
            value = TaskCreate.model_validate(value)
        if type(value) is not TaskCreate:
            raise ValueError("Graph nodes require task creation requests.")
        copied = copy_task_create(value)
        if copied.task_id is None:
            raise ValueError("Graph nodes require explicit task identities.")
        graph_identifier(copied.task_id)
        return copied

    @field_validator("prerequisite_task_ids")
    @classmethod
    def canonical_prerequisites(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        checked = tuple(graph_identifier(item) for item in value)
        if len(set(checked)) != len(checked):
            raise ValueError("Duplicate prerequisites are not allowed.")
        return tuple(sorted(checked))


class TaskGraphCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    graph_id: str
    nodes: tuple[TaskGraphNode, ...] = Field(min_length=1, max_length=TASK_GRAPH_MAX_NODES)
    _submitted_request_sha256: str | None = PrivateAttr(default=None)
    _parent_invocations: tuple[TaskInvocationSnapshot, ...] = PrivateAttr(default=())

    @field_validator("graph_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return graph_identifier(value)

    @field_validator("nodes", mode="before")
    @classmethod
    def copy_nodes(cls, value: object) -> tuple[TaskGraphNode, ...]:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= TASK_GRAPH_MAX_NODES:
            raise ValueError("Graph node count is outside its bound.")
        result = []
        for node in value:
            if type(node) is TaskGraphNode:
                node = TaskGraphNode(
                    task=node.task, prerequisite_task_ids=node.prerequisite_task_ids
                )
            else:
                node = TaskGraphNode.model_validate(node)
            result.append(node)
        return tuple(sorted(result, key=lambda node: node.task.task_id or ""))

    @model_validator(mode="after")
    def validate_graph(self) -> TaskGraphCreate:
        ids = {node.task.task_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("Duplicate graph task identities are not allowed.")
        if sum(len(node.prerequisite_task_ids) for node in self.nodes) > TASK_GRAPH_MAX_EDGES:
            raise ValueError("Graph edge count exceeds its bound.")
        remaining = {node.task.task_id: set(node.prerequisite_task_ids) for node in self.nodes}
        if any(not prerequisites <= ids for prerequisites in remaining.values()):
            raise ValueError("Prerequisites must belong to the same graph submission.")
        while remaining:
            ready = {identity for identity, prerequisites in remaining.items() if not prerequisites}
            if not ready:
                raise ValueError("Task graph contains a cycle.")
            remaining = {
                identity: prerequisites - ready
                for identity, prerequisites in remaining.items()
                if identity not in ready
            }
        if len(
            canonical_durable_json_bytes(self.model_dump(mode="json", warnings=False), "graph")
        ) > (TASK_GRAPH_MAX_BYTES):
            raise ValueError("Task graph exceeds its canonical byte limit.")
        return self


def copy_task_graph_create(request: TaskGraphCreate) -> TaskGraphCreate:
    """Revalidate mutated models without erasing private invocation provenance."""
    if type(request) is not TaskGraphCreate:
        raise TypeError("Graph creation requires a TaskGraphCreate request.")
    copied = TaskGraphCreate(graph_id=request.graph_id, nodes=request.nodes)
    submitted = request._submitted_request_sha256
    if submitted is not None and (
        type(submitted) is not str
        or len(submitted) != 64
        or any(character not in "0123456789abcdef" for character in submitted)
    ):
        raise ValueError("Invalid graph submission authority.")
    parents = request._parent_invocations
    if type(parents) is not tuple or len(parents) > TASK_GRAPH_MAX_NODES:
        raise ValueError("Invalid graph parent authority.")
    if any(type(parent) is not TaskInvocationSnapshot for parent in parents):
        raise ValueError("Invalid graph parent authority.")
    copied._submitted_request_sha256 = submitted
    copied._parent_invocations = tuple(
        TaskInvocationSnapshot.model_validate(parent.model_dump(mode="python", warnings=False))
        for parent in parents
    )
    if tuple(parent.id for parent in copied._parent_invocations) != tuple(
        sorted({parent.id for parent in copied._parent_invocations})
    ):
        raise ValueError("Graph parent authority must be unique and ordered.")
    return copied


def task_graph_with_runtime_admission(
    request: TaskGraphCreate,
    *,
    submitted_request_sha256: str,
    parents: tuple[TaskInvocationSnapshot, ...],
) -> TaskGraphCreate:
    """Carry detached SDK preflight authority into the store's atomic boundary.

    These are internal preparation fields, never accepted from graph request JSON.
    The submitted digest permits read-only replay without resolving context again;
    the admission digest also binds every resolved parent and session authority.
    """
    copied = copy_task_graph_create(request)
    copied._submitted_request_sha256 = submitted_request_sha256
    copied._parent_invocations = parents
    return copy_task_graph_create(copied)


def task_graph_authority_document(request: TaskGraphCreate) -> dict:
    request = copy_task_graph_create(request)
    document = request.model_dump(mode="json", warnings=False)
    document["submitted_request_sha256"] = request._submitted_request_sha256
    document["parent_invocations"] = [
        parent.model_dump(mode="json", warnings=False) for parent in request._parent_invocations
    ]
    # Public equal-shaped claims and authenticated preparation are not equivalent.
    document["invocation_authorities"] = [
        {
            "verified_origin": (
                None
                if node.task._verified_invocation_origin is None
                else node.task._verified_invocation_origin.model_dump(mode="json")
            ),
            "runtime_source": (
                None
                if node.task._runtime_invocation_source is None
                else node.task._runtime_invocation_source.value
            ),
            "session_binding": (
                None
                if node.task._runtime_session_binding is None
                else node.task._runtime_session_binding.model_dump(mode="json")
            ),
        }
        for node in request.nodes
    ]
    return document


def task_graph_request_sha256(request: TaskGraphCreate) -> str:
    canonical = canonical_durable_json_bytes(task_graph_authority_document(request), "graph")
    if len(canonical) > TASK_GRAPH_MAX_BYTES:
        raise ValueError("Task graph authority exceeds its canonical byte limit.")
    return sha256(canonical).hexdigest()


class TaskGraphCreationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    graph_id: GraphIdentifier
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submitted_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_ids: GraphIdentifiers = Field(min_length=1)
    accepted_at: datetime

    @field_validator("accepted_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "accepted_at")


class TaskGraphEventType(StrEnum):
    WAITING_GROUP = "task.group_waiting"
    CREATED = "task.graph_created"
    WAITING = "task.dependency_waiting"
    READY = "task.dependencies_satisfied"
    SKIPPED = "task.dependency_skipped"
    TERMINAL = "task.graph_member_terminal"


class TaskGraphEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    graph_id: GraphIdentifier
    sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    type: TaskGraphEventType
    occurred_at: datetime
    task_id: GraphIdentifier | None = None
    status: TaskStatus | None = None
    prerequisite_task_ids: GraphIdentifiers = ()

    @field_validator("occurred_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @model_validator(mode="after")
    def validate_event(self) -> TaskGraphEvent:
        if self.type is TaskGraphEventType.CREATED:
            if (
                self.sequence != 1
                or self.task_id is not None
                or self.status is not None
                or self.prerequisite_task_ids
            ):
                raise ValueError("Graph creation event has contradictory member evidence.")
        else:
            if self.task_id is None or self.status is None or self.sequence == 1:
                raise ValueError("Graph member event requires member evidence.")
            expected = {
                TaskGraphEventType.WAITING_GROUP: TaskStatus.WAITING_GROUP,
                TaskGraphEventType.WAITING: TaskStatus.WAITING_DEPENDENCIES,
                TaskGraphEventType.READY: TaskStatus.PENDING,
                TaskGraphEventType.SKIPPED: TaskStatus.DEPENDENCY_SKIPPED,
            }.get(self.type)
            if expected is not None and self.status is not expected:
                raise ValueError("Graph event status contradicts its type.")
            if self.type is TaskGraphEventType.TERMINAL and self.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                raise ValueError("Graph terminal event requires a terminal outcome.")
            if (
                self.type in {TaskGraphEventType.WAITING, TaskGraphEventType.SKIPPED}
                and not self.prerequisite_task_ids
            ):
                raise ValueError("Graph dependency event requires prerequisite evidence.")
        return self


class TaskGraphMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: GraphIdentifier
    prerequisite_task_ids: GraphIdentifiers
    status: TaskStatus
    failed_prerequisite_task_ids: GraphIdentifiers = ()

    @model_validator(mode="after")
    def validate_member(self) -> TaskGraphMember:
        if self.task_id in self.prerequisite_task_ids:
            raise ValueError("Graph member cannot depend on itself.")
        if not set(self.failed_prerequisite_task_ids) <= set(self.prerequisite_task_ids):
            raise ValueError("Failed prerequisites must belong to this member.")
        if (self.status is TaskStatus.DEPENDENCY_SKIPPED) != bool(
            self.failed_prerequisite_task_ids
        ):
            raise ValueError("Dependency skip requires exact failed prerequisite evidence.")
        return self


class TaskGraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    receipt: TaskGraphCreationReceipt
    members: tuple[TaskGraphMember, ...] = Field(min_length=1, max_length=TASK_GRAPH_MAX_NODES)
    last_sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    @model_validator(mode="after")
    def validate_snapshot(self) -> TaskGraphSnapshot:
        if tuple(member.task_id for member in self.members) != self.receipt.task_ids:
            raise ValueError("Graph snapshot membership contradicts admission.")
        members = {member.task_id: member for member in self.members}
        if sum(len(member.prerequisite_task_ids) for member in self.members) > TASK_GRAPH_MAX_EDGES:
            raise ValueError("Graph snapshot exceeds the edge bound.")
        remaining = {member.task_id: set(member.prerequisite_task_ids) for member in self.members}
        if any(not dependencies <= members.keys() for dependencies in remaining.values()):
            raise ValueError("Graph snapshot has external prerequisites.")
        while remaining:
            ready = {identity for identity, dependencies in remaining.items() if not dependencies}
            if not ready:
                raise ValueError("Graph snapshot contains a cycle.")
            remaining = {
                identity: dependencies - ready
                for identity, dependencies in remaining.items()
                if identity not in ready
            }
        if self.last_sequence < len(self.members) + 1:
            raise ValueError("Graph snapshot lacks admission event evidence.")
        failures = {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DEPENDENCY_SKIPPED}
        for member in self.members:
            prerequisites = [members[identity] for identity in member.prerequisite_task_ids]
            ready = all(item.status is TaskStatus.COMPLETED for item in prerequisites)
            failed = {item.task_id for item in prerequisites if item.status in failures}
            if (
                member.status
                in {
                    TaskStatus.PENDING,
                    TaskStatus.CLAIMED,
                    TaskStatus.RUNNING,
                    TaskStatus.COMPLETED,
                }
                and not ready
            ):
                raise ValueError("Graph member became executable before its prerequisites.")
            if member.status is TaskStatus.WAITING_DEPENDENCIES and (ready or failed):
                raise ValueError("Graph waiting state contradicts prerequisite outcomes.")
            if not set(member.failed_prerequisite_task_ids) <= failed:
                raise ValueError("Graph skip evidence contradicts prerequisite outcomes.")
            if failed and member.status not in failures | {TaskStatus.COMPLETED}:
                raise ValueError("Failed dependencies have not been propagated.")
        return self
