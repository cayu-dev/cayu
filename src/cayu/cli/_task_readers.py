"""One read seam for `cayu task`: a local task store or a Cayu server's task API.

Both readers return tasks in the shape of the Cayu server's task API (``GET
/api/tasks`` and ``GET /api/tasks/{id}``), so the command renders one shape whichever
it read. ``StoreTaskReader`` takes any ``TaskStore``; ``open_store_task_reader`` opens a
SQLite or Postgres one read-only. ``ApiTaskReader`` takes any ``CayuApi``, such as a Cayu
Cloud Agent's (`open_cloud_agent_api`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli.store_targets import SessionStoreBackend, SessionStoreTarget
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.store import TaskStore

# The server's task list takes at most this many rows per page (`TaskQuery.limit`).
MAX_TASK_PAGE = 1000


@dataclass(frozen=True)
class TaskListRequest:
    """Filters and one page of a newest-updated-first task list."""

    status: TaskStatus | None = None
    type: str | None = None
    agent: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    limit: int = 50
    offset: int = 0


class TaskReader(Protocol):
    """Reads tasks from one source; see the module docstring for the row shape."""

    source: dict[str, Any]

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        """One page of list rows (no ``input``, ``result``, ``error`` or ``metadata``)."""
        ...

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        """The task's detail, with its result, or ``None`` when there is no such task."""
        ...

    async def close(self) -> None: ...


class StoreTaskReader:
    """Reads a ``TaskStore`` the caller opened; ``close`` closes it."""

    def __init__(self, store: TaskStore, *, source: dict[str, Any]) -> None:
        self._store = store
        self.source = source

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        tasks = await self._store.list_tasks(
            TaskQuery(
                status=request.status,
                type=request.type,
                assigned_agent_name=request.agent,
                session_id=request.session_id,
                parent_task_id=request.parent_task_id,
                limit=request.limit,
                offset=request.offset,
                order_by=TaskOrder.UPDATED_AT_DESC,
            )
        )
        return [task_projection(task) for task in tasks]

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        task = await self._store.load_task(task_id)
        return None if task is None else task_projection(task, detail=True)

    async def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            await close()


class ApiTaskReader:
    """Reads a Cayu server's task API; ``close`` closes the API client."""

    def __init__(self, api: CayuApi, *, source: dict[str, Any]) -> None:
        self._api = api
        self.source = source

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        query = {
            key: value
            for key, value in {
                "status": None if request.status is None else request.status.value,
                "type": request.type,
                "assigned_agent_name": request.agent,
                "session_id": request.session_id,
                "parent_task_id": request.parent_task_id,
                "limit": str(request.limit),
                "offset": str(request.offset),
                "order_by": TaskOrder.UPDATED_AT_DESC.value,
            }.items()
            if value is not None
        }
        body = self._api.get("/tasks", query=query, resource="tasks")
        if not isinstance(body, list) or not all(_is_task_row(row) for row in body):
            raise CayuApiError(
                "not_cayu", f"{self._api.description} returned an invalid task list."
            )
        return body

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        try:
            body = self._api.get(f"/tasks/{quote(task_id, safe='')}", resource="tasks")
        except CayuApiNotFound:
            return None
        if not _is_task_row(body):
            raise CayuApiError("not_cayu", f"{self._api.description} returned an invalid task.")
        return body

    async def close(self) -> None:
        self._api.close()


def open_store_task_reader(target: SessionStoreTarget) -> StoreTaskReader:
    """Open the task store in ``target``'s database read-only."""

    from cayu.storage import migrations as schema

    if target.backend is SessionStoreBackend.SQLITE:
        path = target.sqlite_path
        if path is None:
            raise AssertionError("Resolved SQLite target has no path.")
        if not path.is_file():
            raise ValueError(f"No SQLite database exists at {path}.")
        from cayu.storage import SQLiteTaskStore

        store: TaskStore = SQLiteTaskStore(
            path, schema_mode=schema.SchemaMode.VALIDATE, read_only=True
        )
        return StoreTaskReader(store, source={"kind": "sqlite", "path": str(path)})
    if target.postgres_dsn is None:
        raise AssertionError("Resolved Postgres target has no DSN.")
    from cayu.storage import PostgresTaskStore

    store = PostgresTaskStore(
        target.postgres_dsn, schema_mode=schema.SchemaMode.VALIDATE, read_only=True
    )
    return StoreTaskReader(store, source={"kind": "postgres"})


def task_projection(task: Task, *, detail: bool = False) -> dict[str, Any]:
    """A local task in the server's public task shape (list row, or detail)."""

    row: dict[str, Any] = {
        "id": task.id,
        "graph_id": task.graph_id,
        "prerequisite_task_ids": list(task.prerequisite_task_ids),
        "concurrency_key": task.concurrency_key,
        "type": task.type,
        "title": task.title,
        "description": task.description,
        "status": task.status.value,
        "status_reason": task.status_reason,
        "status_payload": task.status_payload,
        "session_id": task.session_id,
        "parent_task_id": task.parent_task_id,
        "assigned_agent_name": task.assigned_agent_name,
        "available_at": _iso(task.available_at),
        "worker_id": task.worker_id,
        "lease_expires_at": _iso(task.lease_expires_at),
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "completed_at": _iso(task.completed_at),
        "retry_series": None if task.retry_series is None else _retry_series(task),
        "schedule": None if task.schedule is None else _schedule(task),
    }
    if detail:
        row.update(
            input=task.input,
            result=task.result,
            error=task.error,
            metadata=task.metadata,
            started_at=_iso(task.started_at),
        )
    return row


def _retry_series(task: Task) -> dict[str, Any]:
    series = task.retry_series
    assert series is not None
    projected = series.model_dump(mode="json", warnings=False, exclude={"authority_sha256"})
    projected["cumulative_tokens"] = str(series.cumulative_tokens)
    projected["tokens_remaining"] = (
        None if series.tokens_remaining is None else str(series.tokens_remaining)
    )
    projected["policy"]["max_total_tokens"] = (
        None if series.policy.max_total_tokens is None else str(series.policy.max_total_tokens)
    )
    return projected


def _schedule(task: Task) -> dict[str, Any]:
    # Like the server, show the controls and times, not the schedule's content hashes.
    schedule = task.schedule
    assert schedule is not None
    return {
        "schema_version": schedule.schema_version,
        "revision": schedule.revision,
        "admitted_at": _iso(schedule.admitted_at),
        "policy": {
            "expires_at": _iso(schedule.policy.expires_at),
            "misfire_policy": schedule.policy.misfire_policy.value,
            "misfire_grace_seconds": schedule.policy.misfire_grace_seconds,
        },
    }


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _is_task_row(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    row = cast("dict[str, Any]", value)
    return all(isinstance(row.get(key), str) for key in ("id", "status", "type"))
