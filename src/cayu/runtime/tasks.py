from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_object, require_clean_nonblank, require_nonblank


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    title: str | None = None
    description: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
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
        return copy_json_object(value, info.field_name)

    @field_validator("result", "error", mode="before")
    @classmethod
    def copy_optional_json_object(
        cls,
        value: dict[str, Any] | None,
        info,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_json_object(value, info.field_name)

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


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        return copy_json_object(value, info.field_name)

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
    model_config = ConfigDict(extra="forbid")

    status: TaskStatus | None = None
    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0)
    order_by: TaskOrder = TaskOrder.UPDATED_AT_DESC

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


class TaskStore(ABC):
    """Persistent store for durable work items."""

    @abstractmethod
    async def create_task(self, request: TaskCreate) -> Task:
        """Create a task."""

    @abstractmethod
    async def load_task(self, task_id: str) -> Task | None:
        """Load a task by id."""

    @abstractmethod
    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        """List tasks for dashboards, queues, and orchestration."""

    @abstractmethod
    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        """Mark a pending task as running."""

    @abstractmethod
    async def complete_task(self, task_id: str, result: dict[str, Any]) -> Task:
        """Mark a pending or running task as completed."""

    @abstractmethod
    async def fail_task(self, task_id: str, error: dict[str, Any]) -> Task:
        """Mark a pending or running task as failed."""

    @abstractmethod
    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a pending or running task as cancelled."""


class InMemoryTaskStore(TaskStore):
    """In-process task store for tests, local development, and examples."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _task_from_create(request)
            if task.id in self._tasks:
                raise ValueError(f"Task already exists: {task.id}")
            self._tasks[task.id] = task
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
            _ensure_can_transition(task, TaskStatus.RUNNING)
            now = datetime.now(UTC)
            updated = task.model_copy(
                update={
                    "status": TaskStatus.RUNNING,
                    "session_id": (session_id if session_id is not None else task.session_id),
                    "started_at": task.started_at or now,
                    "updated_at": now,
                }
            )
            self._tasks[task_id] = updated
            return updated.model_copy(deep=True)

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
            )

    async def fail_task(self, task_id: str, error: dict[str, Any]) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
            )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> Task:
        task = self._require_task(task_id)
        _ensure_can_transition(task, status)
        now = datetime.now(UTC)
        updated = task.model_copy(
            update={
                "status": status,
                "result": deepcopy(result),
                "error": deepcopy(error),
                "started_at": task.started_at or now,
                "completed_at": now,
                "updated_at": now,
            }
        )
        self._tasks[task_id] = updated
        return updated.model_copy(deep=True)


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
        input=copy_json_object(task.input, "input"),
        result=None if task.result is None else copy_json_object(task.result, "result"),
        error=None if task.error is None else copy_json_object(task.error, "error"),
        metadata=copy_json_object(task.metadata, "metadata"),
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
        input=copy_json_object(request.input, "input"),
        metadata=copy_json_object(request.metadata, "metadata"),
    )


def copy_task_query(query: TaskQuery | None) -> TaskQuery:
    if query is None:
        return TaskQuery()
    if type(query) is not TaskQuery:
        raise TypeError("Task queries must be TaskQuery instances.")
    return TaskQuery(
        status=query.status,
        type=query.type,
        session_id=query.session_id,
        parent_task_id=query.parent_task_id,
        assigned_agent_name=query.assigned_agent_name,
        limit=query.limit,
        offset=query.offset,
        order_by=query.order_by,
    )


def _task_from_create(request: TaskCreate) -> Task:
    now = datetime.now(UTC)
    values = {
        "type": request.type,
        "title": request.title,
        "description": request.description,
        "status": TaskStatus.PENDING,
        "session_id": request.session_id,
        "parent_task_id": request.parent_task_id,
        "assigned_agent_name": request.assigned_agent_name,
        "input": copy_json_object(request.input, "input"),
        "metadata": copy_json_object(request.metadata, "metadata"),
        "created_at": now,
        "updated_at": now,
    }
    if request.task_id is not None:
        values["id"] = request.task_id
    return Task(**values)


def _ensure_can_transition(task: Task, next_status: TaskStatus) -> None:
    if task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")
    if next_status == TaskStatus.RUNNING and task.status != TaskStatus.PENDING:
        raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")


def _task_matches(task: Task, query: TaskQuery) -> bool:
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
