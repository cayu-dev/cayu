"""Exact declared creation and source-bound replay contracts."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.tasks.types import TaskTypeSpec, copy_task_type, task_input_validator

if TYPE_CHECKING:
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.records import Task


class TaskCreationConflict(ValueError):
    """The selected creation identity or replay source no longer matches."""


class TaskCreationResultDeleted(TaskCreationConflict):
    """An exact operation already committed, but its task has been deleted."""


class TaskReplaySource(BaseModel):
    """Public incarnation proof; never a digest of private task input."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    task_id: str
    created_at: datetime
    root_invocation_id: str
    restart_count: int = Field(ge=0, strict=True)
    session_id: str | None
    session_instance_id: str | None

    @field_validator("task_id", "root_invocation_id", "session_id", "session_instance_id")
    @classmethod
    def clean_identity(cls, value: str | None) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, "replay identity")


def task_replay_source(task: Task) -> TaskReplaySource:
    return TaskReplaySource(
        task_id=task.id,
        created_at=task.created_at,
        root_invocation_id=task.invocation.root_invocation_id,
        restart_count=task.restart_count,
        session_id=task.session_id,
        session_instance_id=task.session_instance_id,
    )


class TaskCreationOperation(BaseModel):
    """Complete declared intent paired with a caller-stable TaskCreate.task_id."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    operation_id: str = Field(max_length=256)
    declaration: TaskTypeSpec
    replay_source: TaskReplaySource | None = None

    @field_validator("operation_id")
    @classmethod
    def clean_operation(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "operation_id")


def copy_creation_operation(operation: TaskCreationOperation) -> TaskCreationOperation:
    if type(operation) is not TaskCreationOperation:
        raise TypeError("A typed task creation operation is required.")
    source = operation.replay_source
    if source is not None:
        if type(source) is not TaskReplaySource:
            raise TypeError("A typed task replay source is required.")
        source = TaskReplaySource(
            **{name: getattr(source, name) for name in TaskReplaySource.model_fields}
        )
    return TaskCreationOperation(
        operation_id=operation.operation_id,
        declaration=copy_task_type(operation.declaration),
        replay_source=source,
    )


def creation_operation_digest(request: TaskCreate, operation: TaskCreationOperation) -> str:
    from cayu.resource_access import current_binding
    from cayu.tasks._scheduling import schedule_creation_digest
    from cayu.tasks.access import _creation

    if request.task_id is None:
        raise ValueError("An exact task creation requires a stable task_id.")
    if any(
        (
            request.session_id,
            request.parent_task_id,
            request.work_contract,
            request.schedule_policy,
            request.retry_policy,
            request.available_at,
            request.concurrency_key,
            request.restartable,
            request._runtime_session_binding,
        )
    ):
        raise ValueError("Declared creation must create a fresh ordinary root task.")
    if request.type != operation.declaration.name or (
        request.assigned_agent_name != operation.declaration.agent_name
    ):
        raise TaskCreationConflict("Task creation does not match its declared type.")
    if operation.replay_source is not None and request.input:
        raise TaskCreationConflict("Task replay does not accept input overrides.")
    binding = current_binding()
    access = _creation.get()
    # The existing scheduling fingerprint already includes every TaskCreate
    # field and private provenance. Reuse it rather than maintain a second set.
    document = {
        "creation": schedule_creation_digest(request),
        "operation": operation.model_dump(mode="json", warnings="error"),
        "resource_binding": None if binding is None else binding.model_dump(mode="json"),
        "access_labels": None if access is None else dict(access.labels),
    }
    return sha256(canonical_durable_json_bytes(document, "task creation operation")).hexdigest()


def prepare_creation_operation(
    request: TaskCreate,
    operation: TaskCreationOperation,
    source: Task | None,
) -> TaskCreate:
    from cayu.tasks.access import _collection_bounds, require_read
    from cayu.tasks.creation import copy_task_create

    prepared = copy_task_create(request)
    if operation.replay_source is not None:
        if source is not None and (
            (source.session_id is None) != (source.session_instance_id is None)
        ):
            raise TaskCreationConflict("Task replay source lacks exact attachment identity.")
        if source is None or task_replay_source(source) != operation.replay_source:
            raise TaskCreationConflict("Task replay source changed or is unavailable.")
        bounds = _collection_bounds.get()
        if bounds is not None:
            require_read(source, bounds)
        if source.type != operation.declaration.name or (
            source.assigned_agent_name != operation.declaration.agent_name
        ):
            raise TaskCreationConflict("Task replay source does not match the declared type.")
        prepared.input = source.input
        prepared = copy_task_create(prepared)
    try:
        valid = task_input_validator(operation.declaration).is_valid(prepared.input)
    except Exception:
        raise ValueError("Task input schema cannot validate this input.") from None
    if not valid:
        raise ValueError("Task input does not match its declared schema.")
    return prepared


def verify_creation_receipt(
    retained_digest: str,
    expected_digest: str | None,
    task: Task | None,
) -> Task:
    if retained_digest != expected_digest:
        raise TaskCreationConflict("Task creation operation identity has different content.")
    if task is None:
        raise TaskCreationResultDeleted("The task created by this operation was deleted.")
    return task.model_copy(deep=True)


def checked_replay_sources(tasks: list[Task]) -> list[Task]:
    from cayu.tasks.access import _collection_bounds, require_read

    bounds = _collection_bounds.get()
    if bounds is not None:
        for task in tasks:
            require_read(task, bounds)
    return [task.model_copy(deep=True) for task in tasks]
