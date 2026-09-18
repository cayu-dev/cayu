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
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
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


class TaskGroupQuiescencePolicy(BaseModel):
    """Opt-in bounded observation; expiry never proves that effects stopped."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    timeout_seconds: StrictFloat | StrictInt = Field(gt=0, le=86_400, allow_inf_nan=False)


class TaskGroupQuiescenceStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    WAITING_DECISION = "waiting_decision"
    DRAINING = "draining"
    ATTENTION_REQUIRED = "attention_required"
    QUIESCENT = "quiescent"


class TaskGroupFinalizerStatus(StrEnum):
    ABSENT = "absent"
    WAITING = "waiting"
    RELEASED = "released"
    INELIGIBLE = "ineligible"
    SETTLED = "settled"


class TaskGroupInvocationObligation(BaseModel):
    """Exact ownerless invocation; worker callbacks retain their separate owner."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: GraphIdentifier
    session_id: str = Field(strict=True)
    session_instance_id: GraphIdentifier
    interaction_id: GraphIdentifier
    run_epoch: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    profile_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_record_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    owner_settled: bool = Field(default=False, strict=True)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        from cayu.sessions.base import _require_bounded_session_id

        return _require_bounded_session_id(
            require_durable_clean_nonblank(value, "session_id"), "session_id"
        )

    @model_validator(mode="after")
    def validate_release_owner(self):
        if self.release_record_sha256 is not None and not self.owner_settled:
            raise ValueError("Invocation release requires settled execution ownership.")
        return self


class TaskGroupResultResolutionPending(TaskGroupConflict):
    """A retained result callback still owns the group's execution barrier."""


class TaskGroupResultResolutionObligation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    decision_id: str = Field(strict=True)
    owner_id: str = Field(strict=True, pattern=r"^[0-9a-f]{32}$")
    settled_at: datetime | None = None

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, value: str) -> str:
        from cayu.tasks.contracts import validate_work_completion_linked_id

        return validate_work_completion_linked_id(value, "decision_id")

    @field_validator("settled_at")
    @classmethod
    def normalize_settlement(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "settled_at")


class TaskGroupExecutionObligation(BaseModel):
    """Execution evidence survives terminal task publication and lease clearing."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: GraphIdentifier
    worker_id: str | None = Field(default=None, strict=True)
    started_at: datetime
    settled_at: datetime | None = None
    invocation: TaskGroupInvocationObligation | None = None
    result_resolution: TaskGroupResultResolutionObligation | None = None

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str | None) -> str | None:
        # This is TaskStore worker authority, not a graph-member identifier.
        return None if value is None else require_clean_nonblank(value, "worker_id")

    @field_validator("started_at", "settled_at")
    @classmethod
    def normalize_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "execution time")

    @model_validator(mode="after")
    def validate_result_settlement(self):
        if (
            self.settled_at is not None
            and self.result_resolution is not None
            and self.result_resolution.settled_at is None
        ):
            raise ValueError("Execution cannot settle before its result resolver.")
        return self


class TaskGroupQuiescence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status: TaskGroupQuiescenceStatus = TaskGroupQuiescenceStatus.NOT_REQUESTED
    deadline: datetime | None = None
    loser_task_ids: GraphIdentifiers = ()
    unsettled_task_ids: GraphIdentifiers = ()
    # At most 128 selected roots, each with at most 100 automatic attempts.
    executions: tuple[TaskGroupExecutionObligation, ...] = Field(default=(), max_length=12_800)
    finalizer_status: TaskGroupFinalizerStatus = TaskGroupFinalizerStatus.ABSENT

    @field_validator("deadline")
    @classmethod
    def normalize_deadline(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "deadline")

    @model_validator(mode="after")
    def validate_evidence(self) -> TaskGroupQuiescence:
        if not set(self.unsettled_task_ids) <= set(self.loser_task_ids):
            raise ValueError("Unsettled group work is outside its loser scope.")
        ids = tuple(item.task_id for item in self.executions)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Group execution obligations must be unique and ordered.")
        if self.status in {
            TaskGroupQuiescenceStatus.NOT_REQUESTED,
            TaskGroupQuiescenceStatus.WAITING_DECISION,
        }:
            if self.deadline is not None or self.loser_task_ids or self.unsettled_task_ids:
                raise ValueError("Undecided group cannot carry a draining deadline.")
        elif self.deadline is None:
            raise ValueError("Decided quiescence requires a durable deadline.")
        if self.status is TaskGroupQuiescenceStatus.QUIESCENT and self.unsettled_task_ids:
            raise ValueError("Quiescent group retains unsettled work.")
        return self


class TaskGroupQuiescenceResolution(BaseModel):
    """Explicit post-timeout authorization; never substitutes for settlement proof."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: GraphIdentifier
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    idempotency_key: GraphIdentifier


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
    quiescence: TaskGroupQuiescencePolicy | None = None
    finalizer_task_id: GraphIdentifier | None = None
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
        if self.finalizer_task_id is not None:
            if self.quiescence is None:
                raise ValueError("A group finalizer requires a quiescence policy.")
            dependencies = {
                node.task.task_id: node.prerequisite_task_ids for node in self.graph.nodes
            }
            if (
                self.finalizer_task_id not in dependencies
                or self.finalizer_task_id in self.member_task_ids
            ):
                raise ValueError("The finalizer must be a nonmember in the submitted graph.")

            def ancestors(identity: str) -> set[str]:
                result: set[str] = set()
                pending = list(dependencies[identity])
                while pending:
                    parent = pending.pop()
                    if parent not in result:
                        result.add(parent)
                        pending.extend(dependencies[parent])
                return result

            if ancestors(self.finalizer_task_id) & set(self.member_task_ids) or any(
                self.finalizer_task_id in ancestors(identity) for identity in self.member_task_ids
            ):
                raise ValueError("Finalizer dependencies cannot cross selected group members.")
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
        quiescence=None
        if request.quiescence is None
        else TaskGroupQuiescencePolicy(timeout_seconds=request.quiescence.timeout_seconds),
        finalizer_task_id=request.finalizer_task_id,
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
    quiescence: TaskGroupQuiescencePolicy | None = None
    finalizer_task_id: GraphIdentifier | None = None
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submitted_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_membership(self) -> TaskGroupCreationReceipt:
        if self.finalizer_task_id is not None and (
            self.quiescence is None
            or self.finalizer_task_id not in self.graph.task_ids
            or self.finalizer_task_id in self.member_task_ids
        ):
            raise ValueError("Group receipt has invalid finalizer authority.")
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
    quiescence: TaskGroupQuiescence = Field(default_factory=TaskGroupQuiescence)
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
        minimum_sequence = 1 + terminal_count + (2 if self.decision else 0)
        if self.last_sequence < minimum_sequence or (
            self.receipt.quiescence is None and self.last_sequence != minimum_sequence
        ):
            raise ValueError("Group event cursor contradicts terminal evidence.")
        barrier = self.quiescence
        if (self.receipt.quiescence is None) != (
            barrier.status is TaskGroupQuiescenceStatus.NOT_REQUESTED
        ):
            raise ValueError("Group quiescence contradicts its admission policy.")
        if (self.receipt.finalizer_task_id is None) != (
            barrier.finalizer_status is TaskGroupFinalizerStatus.ABSENT
        ):
            raise ValueError("Group finalizer contradicts its admission authority.")
        if not set(barrier.loser_task_ids) <= set(self.receipt.member_task_ids):
            raise ValueError("Group loser scope contradicts membership.")
        if self.decision is None and barrier.status not in {
            TaskGroupQuiescenceStatus.NOT_REQUESTED,
            TaskGroupQuiescenceStatus.WAITING_DECISION,
        }:
            raise ValueError("Undecided group has a decided barrier.")
        if barrier.finalizer_status in {
            TaskGroupFinalizerStatus.RELEASED,
            TaskGroupFinalizerStatus.SETTLED,
        } and (
            self.decision is None
            or self.decision.status is not TaskGroupStatus.SUCCEEDED
            or barrier.status is not TaskGroupQuiescenceStatus.QUIESCENT
        ):
            raise ValueError("Released finalizer lacks successful quiescence.")
        return self


class TaskGroupEventType(StrEnum):
    CREATED = "task.group_created"
    MEMBER_TERMINAL = "task.group_member_terminal"
    POLICY_SATISFIED = "task.group_policy_satisfied"
    POLICY_IMPOSSIBLE = "task.group_policy_impossible"
    SUCCEEDED = "task.group_succeeded"
    FAILED = "task.group_failed"
    CANCELLATION_REQUESTED = "task.group_cancellation_requested"
    DRAINING = "task.group_draining"
    QUIESCENT = "task.group_quiescent"
    TIMEOUT = "task.group_quiescence_timeout"
    RESOLVED = "task.group_quiescence_resolved"
    FINALIZER_RELEASED = "task.group_finalizer_released"
    FINALIZER_INELIGIBLE = "task.group_finalizer_ineligible"
    FINALIZER_SETTLED = "task.group_finalizer_settled"


class TaskGroupEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    group_id: GraphIdentifier
    sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    type: TaskGroupEventType
    occurred_at: datetime
    task_id: GraphIdentifier | None = None
    task_status: TaskStatus | None = None
    decision: TaskGroupDecision | None = None
    quiescence: TaskGroupQuiescence | None = None

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
        elif self.type in {
            TaskGroupEventType.POLICY_SATISFIED,
            TaskGroupEventType.POLICY_IMPOSSIBLE,
            TaskGroupEventType.SUCCEEDED,
            TaskGroupEventType.FAILED,
        }:
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
        elif self.sequence == 1 or self.quiescence is None or self.decision is not None:
            raise ValueError("Invalid group quiescence event.")
        return self
