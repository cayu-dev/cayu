"""Durable completion policies over an immutable subset of a new task graph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import MAX_PORTABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.tasks.base import TaskStatus
from cayu.tasks.graphs import (
    TASK_GRAPH_MAX_BYTES,
    GraphIdentifier,
    GraphIdentifiers,
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphMember,
    copy_task_graph_create,
)


class TaskGroupConflict(ValueError):
    """The request conflicts with immutable group authority."""


class TaskGroupUnavailable(ValueError):
    """Complete group evidence is unavailable or contradictory."""


class TaskGroupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["all", "first_success", "quorum"]
    k: StrictInt | None = Field(default=None, ge=1, le=128)

    @model_validator(mode="after")
    def validate_policy(self) -> TaskGroupPolicy:
        if (self.kind == "quorum") != (self.k is not None):
            raise ValueError("Only quorum requires a threshold.")
        return self

    def required_successes(self, member_count: int) -> int:
        if self.kind == "all":
            return member_count
        if self.kind == "first_success":
            return 1
        assert self.k is not None
        return self.k


class TaskGroupCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: GraphIdentifier
    graph: TaskGraphCreate
    member_task_ids: GraphIdentifiers = Field(min_length=1)
    policy: TaskGroupPolicy
    _submitted_request_sha256: str | None = PrivateAttr(default=None)

    @field_validator("graph", mode="before")
    @classmethod
    def copy_graph(cls, value: object) -> TaskGraphCreate:
        if isinstance(value, dict):
            value = TaskGraphCreate.model_validate(value)
        if type(value) is not TaskGraphCreate:
            raise ValueError("Group admission requires a task graph.")
        return copy_task_graph_create(value)

    @field_validator("member_task_ids", mode="before")
    @classmethod
    def canonical_members(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 128:
            raise ValueError("Group member count is outside its bound.")
        if any(type(identity) is not str for identity in value):
            raise ValueError("Group member identity must be a string.")
        if len(set(value)) != len(value):
            raise ValueError("Group members must be unique.")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_group(self) -> TaskGroupCreate:
        if not set(self.member_task_ids) <= {node.task.task_id for node in self.graph.nodes}:
            raise ValueError("Group members must belong to the submitted graph.")
        if self.policy.required_successes(len(self.member_task_ids)) > len(self.member_task_ids):
            raise ValueError("Group quorum exceeds membership.")
        document = self.model_dump(mode="json", warnings=False)
        if len(canonical_durable_json_bytes(document, "task group")) > TASK_GRAPH_MAX_BYTES:
            raise ValueError("Task group exceeds its canonical byte bound.")
        return self


def copy_task_group_create(request: TaskGroupCreate) -> TaskGroupCreate:
    if type(request) is not TaskGroupCreate or type(request.policy) is not TaskGroupPolicy:
        raise ValueError("Group admission requires typed authority.")
    copied = TaskGroupCreate(
        group_id=request.group_id,
        graph=request.graph,
        member_task_ids=request.member_task_ids,
        policy=TaskGroupPolicy(kind=request.policy.kind, k=request.policy.k),
    )
    digest = request._submitted_request_sha256
    if digest is not None and (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid group submission digest.")
    copied._submitted_request_sha256 = digest
    return copied


def task_group_request_sha256(request: TaskGroupCreate) -> str:
    request = copy_task_group_create(request)
    # Check the aggregate including private resolved graph authority, not only
    # the public model. A digest alone would conceal an oversized graph envelope.
    from cayu.tasks.graphs import task_graph_authority_document

    document = request.model_dump(mode="json", warnings=False)
    document["graph"] = task_graph_authority_document(request.graph)
    data = canonical_durable_json_bytes(document, "task group authority")
    if len(data) > TASK_GRAPH_MAX_BYTES:
        raise ValueError("Task group authority exceeds its canonical byte bound.")
    return sha256(data).hexdigest()


class TaskGroupStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskGroupCreationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: GraphIdentifier
    graph: TaskGraphCreationReceipt
    member_task_ids: GraphIdentifiers = Field(min_length=1)
    policy: TaskGroupPolicy
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submitted_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_membership(self) -> TaskGroupCreationReceipt:
        if not set(self.member_task_ids) <= set(self.graph.task_ids):
            raise ValueError("Group receipt has foreign members.")
        if self.policy.required_successes(len(self.member_task_ids)) > len(self.member_task_ids):
            raise ValueError("Group receipt has an impossible threshold.")
        return self


class TaskGroupDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: Literal[TaskGroupStatus.SUCCEEDED, TaskGroupStatus.FAILED]
    decided_at: datetime
    successful_task_ids: GraphIdentifiers
    unsuccessful_task_ids: GraphIdentifiers
    reason: Literal["completion_policy_impossible"] | None = None

    @field_validator("decided_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "decided_at")

    @model_validator(mode="after")
    def validate_decision(self) -> TaskGroupDecision:
        if (self.status is TaskGroupStatus.FAILED) != (self.reason is not None):
            raise ValueError("Group decision reason contradicts its outcome.")
        if set(self.successful_task_ids) & set(self.unsuccessful_task_ids):
            raise ValueError("Group decision has contradictory member outcomes.")
        return self


class TaskGroupSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    receipt: TaskGroupCreationReceipt
    members: tuple[TaskGraphMember, ...] = Field(min_length=1, max_length=128)
    decision: TaskGroupDecision | None = None
    last_sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    @property
    def status(self) -> TaskGroupStatus:
        return TaskGroupStatus.PENDING if self.decision is None else self.decision.status

    @model_validator(mode="after")
    def validate_snapshot(self) -> TaskGroupSnapshot:
        if tuple(member.task_id for member in self.members) != self.receipt.member_task_ids:
            raise ValueError("Group snapshot contradicts its membership.")
        if any(
            not set(member.prerequisite_task_ids) <= set(self.receipt.graph.task_ids)
            for member in self.members
        ):
            raise ValueError("Group snapshot has prerequisites outside its graph.")
        failures = {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DEPENDENCY_SKIPPED}
        successes = {m.task_id for m in self.members if m.status is TaskStatus.COMPLETED}
        failed = {m.task_id for m in self.members if m.status in failures}
        required = self.receipt.policy.required_successes(len(self.members))
        if self.decision is None:
            if len(successes) >= required or len(self.members) - len(failed) < required:
                raise ValueError("Decisive group lacks a durable decision.")
        else:
            decision = self.decision
            if (
                not set(decision.successful_task_ids) <= successes
                or not set(decision.unsuccessful_task_ids) <= failed
            ):
                raise ValueError("Group decision contradicts member evidence.")
            if decision.status is TaskGroupStatus.SUCCEEDED:
                if len(decision.successful_task_ids) != required:
                    raise ValueError("Group decision lacks exact contributing successes.")
            elif len(self.members) - len(decision.unsuccessful_task_ids) >= required:
                raise ValueError("Group failure lacks impossibility evidence.")
        terminal_count = len(successes) + len(failed)
        if self.last_sequence != 1 + terminal_count + (2 if self.decision else 0):
            raise ValueError("Group event cursor contradicts terminal evidence.")
        return self


class TaskGroupEventType(StrEnum):
    CREATED = "task.group_created"
    MEMBER_TERMINAL = "task.group_member_terminal"
    POLICY_SATISFIED = "task.group_policy_satisfied"
    POLICY_IMPOSSIBLE = "task.group_policy_impossible"
    SUCCEEDED = "task.group_succeeded"
    FAILED = "task.group_failed"


class TaskGroupEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: GraphIdentifier
    sequence: StrictInt = Field(ge=1, le=131)
    type: TaskGroupEventType
    occurred_at: datetime
    task_id: GraphIdentifier | None = None
    task_status: TaskStatus | None = None
    decision: TaskGroupDecision | None = None

    @field_validator("occurred_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @model_validator(mode="after")
    def validate_event(self) -> TaskGroupEvent:
        if self.type is TaskGroupEventType.CREATED:
            if self.sequence != 1 or any(
                v is not None for v in (self.task_id, self.task_status, self.decision)
            ):
                raise ValueError("Invalid group creation event.")
        elif self.type is TaskGroupEventType.MEMBER_TERMINAL:
            if (
                self.sequence == 1
                or self.task_id is None
                or self.task_status
                not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                    TaskStatus.DEPENDENCY_SKIPPED,
                }
                or self.decision is not None
            ):
                raise ValueError("Invalid terminal group member event.")
        else:
            expected = (
                TaskGroupStatus.SUCCEEDED
                if self.type in {TaskGroupEventType.SUCCEEDED, TaskGroupEventType.POLICY_SATISFIED}
                else TaskGroupStatus.FAILED
            )
            if (
                self.sequence == 1
                or self.task_id is not None
                or self.task_status is not None
                or self.decision is None
                or self.decision.status is not expected
                or self.decision.decided_at != self.occurred_at
            ):
                raise ValueError("Invalid group decision event.")
        return self
