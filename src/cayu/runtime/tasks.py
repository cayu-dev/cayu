from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import copy_json_object, require_clean_nonblank, require_nonblank
from cayu.runtime.aggregates import EXACT_AGGREGATE, AggregateAccuracy


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
        return copy_json_object(value, info.field_name)

    @field_validator("status_payload", "result", "error", mode="before")
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

    q: str | None = None
    status: TaskStatus | None = None
    type: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    assigned_agent_name: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0)
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

    model_config = ConfigDict(extra="forbid")

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

    pending: StrictInt = Field(ge=0)
    claimed: StrictInt = Field(ge=0)
    running: StrictInt = Field(ge=0)
    paused: StrictInt = Field(ge=0)
    blocked: StrictInt = Field(ge=0)
    needs_attention: StrictInt = Field(ge=0)
    completed: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)
    cancelled: StrictInt = Field(ge=0)


class TaskOperationalSnapshot(BaseModel):
    """Exact current task counts captured by one store-local read snapshot."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    total_count: StrictInt = Field(ge=0)
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


class TaskStore(ABC):
    """Persistent store for durable work items."""

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
        """Attach a live worker-claimed task to a session and mark it running."""

    @abstractmethod
    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as completed.

        If ``worker_id`` is given, the update fails unless that worker still owns an active
        lease on the task, so a worker that lost its lease cannot clobber a task another
        worker has since reclaimed.
        """

    @abstractmethod
    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        """Mark a pending or running task as failed.

        If ``worker_id`` is given, the update fails unless that worker still owns an active
        lease on the task.
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
        """Extend the active lease for a claimed or running task owned by ``worker_id``."""

    @abstractmethod
    async def release_task(self, task_id: str, worker_id: str) -> Task:
        """Release a claimed task back to pending and clear worker ownership."""

    @abstractmethod
    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        """Release worker ownership while preserving a running task's session link."""

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

    async def create_running_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _running_task_from_create(request)
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
            self._tasks[task_id] = updated
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
            self._tasks[task_id] = updated
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
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
        error = copy_json_object(error, "error")
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
        copied_error = None if error is None else copy_json_object(error, "error")
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
            self._tasks[task_id] = updated
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
            self._tasks[task.id] = updated
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
            self._tasks[task_id] = updated
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
            self._tasks[task_id] = updated
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
            self._tasks[task_id] = updated
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
                self._tasks[task.id] = updated
                reclaimed.append(updated.model_copy(deep=True))
            return reclaimed

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _require_owned_leased_task(self, task_id: str, worker_id: str) -> Task:
        task = self._require_task(task_id)
        if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
            raise ValueError(f"Task {task.id} is not claimed or running.")
        if task.worker_id != worker_id:
            raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
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
                raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
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
        self._tasks[task_id] = updated
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
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        status_reason=task.status_reason,
        status_payload=(
            None
            if task.status_payload is None
            else copy_json_object(task.status_payload, "status_payload")
        ),
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
        input=copy_json_object(request.input, "input"),
        metadata=copy_json_object(request.metadata, "metadata"),
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
        raise ValueError(f"Task {task.id} has no active lease.")
    if task.lease_expires_at <= now:
        raise ValueError(f"Task {task.id} lease for worker {worker_id} has expired.")


def _raise_task_claim_attach_error(
    task: Task,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> None:
    if task.status is TaskStatus.RUNNING:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        raise ValueError(f"Task {task.id} is already running.")
    if task.status is TaskStatus.CLAIMED:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.worker_id != worker_id:
            raise ValueError(f"Worker {worker_id} does not own task {task.id}.")
        _ensure_active_task_lease(task, worker_id, now=now)
        raise ValueError(f"Task {task.id} is already claimed.")
    _ensure_not_terminal(task)
    raise ValueError(f"Task {task.id} is not claimed by worker {worker_id}.")


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
    return copy_json_object(value, "payload")


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
