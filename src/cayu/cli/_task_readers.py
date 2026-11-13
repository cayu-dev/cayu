"""One read seam for `cayu task`: a local task store or a Cayu server's task API.

Both readers return tasks in the shape of the Cayu server's task API (``GET
/api/tasks`` and ``GET /api/tasks/{id}``), so the command renders one shape whichever
it read. They also read the sessions run under a task (``SessionQuery.task_id``): their
time breakdown and the event-loop lag recorded on them. ``StoreTaskReader`` takes any
``TaskStore``; ``open_store_task_reader`` opens a SQLite or Postgres one read-only.
``ApiTaskReader`` takes any ``CayuApi``, such as a Cayu Cloud Agent's
(`open_cloud_agent_api`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, cast
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli.store_targets import SessionStoreBackend, SessionStoreTarget
from cayu.sessions._event_loop_lag import read_sessions_event_loop_lag
from cayu.sessions._rollup import session_timing
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import EventRecord
from cayu.sessions.store import SessionStore
from cayu.tasks.index import TaskCountsCheck
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import Task, TaskStatus, task_error_code, task_error_summary
from cayu.tasks.store import TaskStore

# The server's task list takes at most this many rows per page (`TaskQuery.limit`).
MAX_TASK_PAGE = 1000
# Sessions run under one task that `cayu task show` reads; the rest are left out.
MAX_TASK_SESSIONS = 100
# A session is created after the task it runs under; this allows for clock skew.
_TASK_SESSION_SKEW = timedelta(minutes=5)


@dataclass(frozen=True)
class TaskListRequest:
    """Filters and one page of a newest-updated-first task list."""

    status: TaskStatus | None = None
    # One task type; give ``type`` or ``types``, not both.
    type: str | None = None
    # Tasks of any of these types; empty, with no ``type``, means every type.
    types: tuple[str, ...] = ()
    agent: str | None = None
    session_id: str | None = None
    parent_task_id: str | None = None
    limit: int = 50
    offset: int = 0

    def __post_init__(self) -> None:
        if self.type is not None and self.types:
            raise ValueError("Give type or types, not both.")

    @property
    def matched_types(self) -> tuple[str, ...]:
        """The task types listed; empty means every type."""
        return self.types or (() if self.type is None else (self.type,))


@dataclass(frozen=True)
class TaskLatestRequest:
    """The newest finished tasks, or with ``by`` the newest per input key."""

    types: tuple[str, ...] = ()
    status: TaskStatus | None = TaskStatus.COMPLETED
    by: str | None = None
    agent: str | None = None
    limit: int = 1


class TaskReader(Protocol):
    """Reads tasks from one source; see the module docstring for the row shape."""

    source: dict[str, Any]

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        """One page of list rows: no ``input``, ``result``, ``error`` or ``metadata``, but
        the error's ``error_code`` and ``error_summary`` (missing from servers that
        predate them)."""
        ...

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        """The task's detail, with its result, or ``None`` when there is no such task."""
        ...

    async def latest_tasks(self, request: TaskLatestRequest) -> list[dict[str, Any]]:
        """The newest finished tasks (``TaskStore.latest_tasks``), as task details."""
        ...

    async def check_task_counts(self) -> dict[str, Any]:
        """``TaskStore.check_task_counts`` as JSON (``TaskCountsCheck``)."""
        ...

    async def task_session_ids(self, task: dict[str, Any]) -> tuple[list[str], bool] | None:
        """The sessions run under the task, oldest first: at most ``MAX_TASK_SESSIONS``,
        and whether more were left out. ``None`` when the source can't list them."""
        ...

    async def session_timing(self, session_id: str) -> dict[str, Any] | None:
        """The session's time breakdown (``SessionTiming``), or ``None`` when the source
        can't say."""
        ...

    async def event_loop_lag(self, session_ids: Sequence[str]) -> dict[str, Any] | None:
        """The worker's event-loop lag recorded on these sessions
        (``read_sessions_event_loop_lag``), or ``None`` when the source can't say."""
        ...

    async def close(self) -> None: ...


class StoreTaskReader:
    """Reads a ``TaskStore`` the caller opened, and its sessions from ``sessions`` when
    given; ``close`` closes both."""

    def __init__(
        self, store: TaskStore, *, source: dict[str, Any], sessions: SessionStore | None = None
    ) -> None:
        self._store = store
        self._sessions = sessions
        self.source = source

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        tasks = await self._store.list_tasks(
            TaskQuery(
                status=request.status,
                types=request.matched_types or None,
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

    async def check_task_counts(self) -> dict[str, Any]:
        return (await self._store.check_task_counts()).model_dump(mode="json")

    async def latest_tasks(self, request: TaskLatestRequest) -> list[dict[str, Any]]:
        tasks = await self._store.latest_tasks(
            TaskQuery(
                types=request.types or None,
                status=request.status,
                assigned_agent_name=request.agent,
                limit=request.limit,
            ),
            by=request.by,
        )
        return [task_projection(task, detail=True) for task in tasks]

    async def task_session_ids(self, task: dict[str, Any]) -> tuple[list[str], bool] | None:
        if self._sessions is None:
            return None
        ids: list[str] = []
        query = task_session_query(task)
        while True:
            page = await self._sessions.list_sessions(query)
            ids.extend(session.id for session in page.sessions)
            if len(ids) > MAX_TASK_SESSIONS or page.next_cursor is None:
                return ids[:MAX_TASK_SESSIONS], len(ids) > MAX_TASK_SESSIONS
            query = query.model_copy(update={"cursor": page.next_cursor})

    async def session_timing(self, session_id: str) -> dict[str, Any] | None:
        if self._sessions is None or await self._sessions.load(session_id) is None:
            return None
        return (await session_timing(self._sessions, session_id)).model_dump(mode="json")

    async def event_loop_lag(self, session_ids: Sequence[str]) -> dict[str, Any] | None:
        if self._sessions is None:
            return None
        return await read_sessions_event_loop_lag(self._sessions, session_ids)

    async def close(self) -> None:
        for store in (self._store, self._sessions):
            close = getattr(store, "close", None)
            if close is not None:
                await close()


class ApiTaskReader:
    """Reads a Cayu server's task API; ``close`` closes the API client."""

    def __init__(self, api: CayuApi, *, source: dict[str, Any]) -> None:
        self._api = api
        self.source = source

    async def list_tasks(self, request: TaskListRequest) -> list[dict[str, Any]]:
        types = request.matched_types
        query: dict[str, str | list[str]] = {
            key: value
            for key, value in {
                **_type_query(types),
                "status": None if request.status is None else request.status.value,
                "assigned_agent_name": request.agent,
                "session_id": request.session_id,
                "parent_task_id": request.parent_task_id,
                "limit": str(request.limit),
                "offset": str(request.offset),
                "order_by": TaskOrder.UPDATED_AT_DESC.value,
            }.items()
            if value is not None
        }
        body = self._task_list(self._api.get("/tasks", query=query, resource="tasks"))
        if types and any(row["type"] not in types for row in body):
            # A server that predates `types` ignores it and lists every type.
            raise CayuApiError(
                "unsupported",
                f"{self._api.description} can't list several task types at once; "
                "list one --type at a time.",
            )
        return body

    async def latest_tasks(self, request: TaskLatestRequest) -> list[dict[str, Any]]:
        query: dict[str, str | list[str]] = {
            key: value
            for key, value in {
                **_type_query(request.types),
                "status": None if request.status is None else request.status.value,
                "by": request.by,
                "assigned_agent_name": request.agent,
                "limit": str(request.limit),
            }.items()
            if value is not None
        }
        try:
            body = self._api.get("/tasks/latest", query=query, resource="tasks")
        except CayuApiNotFound:
            # A server that predates the route reads `latest` as a task id.
            raise CayuApiError(
                "unsupported",
                f"{self._api.description} can't read the latest tasks; its Cayu predates "
                "`GET /tasks/latest`.",
            ) from None
        return self._task_list(body)

    async def check_task_counts(self) -> dict[str, Any]:
        try:
            body = self._api.get("/tasks/counts/check", resource="tasks")
        except CayuApiNotFound:
            raise CayuApiError(
                "unsupported",
                f"{self._api.description} can't check task counts; its Cayu predates "
                "`GET /tasks/counts/check`.",
            ) from None
        try:
            return TaskCountsCheck.model_validate(body).model_dump(mode="json")
        except ValueError:
            raise CayuApiError(
                "not_cayu", f"{self._api.description} returned invalid task counts."
            ) from None

    def _task_list(self, body: object) -> list[dict[str, Any]]:
        if not isinstance(body, list) or not all(_is_task_row(row) for row in body):
            raise CayuApiError(
                "not_cayu", f"{self._api.description} returned an invalid task list."
            )
        return cast("list[dict[str, Any]]", body)

    async def load_task(self, task_id: str) -> dict[str, Any] | None:
        try:
            body = self._api.get(f"/tasks/{quote(task_id, safe='')}", resource="tasks")
        except CayuApiNotFound:
            return None
        if not _is_task_row(body):
            raise CayuApiError("not_cayu", f"{self._api.description} returned an invalid task.")
        return body

    async def task_session_ids(self, task: dict[str, Any]) -> tuple[list[str], bool] | None:
        from cayu.cli._agent_session_reader import AgentSessionReader

        try:
            page = await AgentSessionReader(self._api).list_sessions(task_session_query(task))
        except CayuApiError as exc:
            # A server without session reads, an Agent whose Cayu predates the task filter,
            # or one that didn't answer just now.
            if exc.category in {"unsupported", "not_found"} or exc.transient:
                return None
            raise
        ids = [row["id"] for row in page.rows if isinstance(row.get("id"), str)]
        return ids[:MAX_TASK_SESSIONS], len(ids) > MAX_TASK_SESSIONS

    async def session_timing(self, session_id: str) -> dict[str, Any] | None:
        from cayu.cli._agent_session_reader import AgentSessionReader

        try:
            return AgentSessionReader(self._api).timing(session_id)
        except ValueError:
            # The task names a session the server no longer has.
            return None
        except CayuApiError as exc:
            if not exc.transient:
                raise
            # The server didn't answer just now; the task itself was read.
            return None

    async def event_loop_lag(self, session_ids: Sequence[str]) -> dict[str, Any] | None:
        from cayu.cli._agent_session_reader import AgentSessionReader

        try:
            return await read_sessions_event_loop_lag(
                _SkipMissingSessions(AgentSessionReader(self._api)), session_ids
            )
        except CayuApiError as exc:
            # An Agent whose Cayu predates the record refuses its event type; the server
            # may also not have answered just now. The task itself was read.
            if exc.category in {"request_rejected", "unsupported"} or exc.transient:
                return None
            raise

    async def close(self) -> None:
        self._api.close()


class _SkipMissingSessions:
    """Reads events as ``reader`` does, finding none for a session the server no longer
    has."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    async def query_events(self, query: EventQuery) -> list[EventRecord]:
        try:
            return await self._reader.query_events(query)
        except ValueError:
            return []


def task_session_query(task: dict[str, Any]) -> SessionQuery:
    """The first page of the sessions run under ``task``, oldest first."""

    created = _instant(task.get("created_at"))
    return SessionQuery(
        task_id=task["id"],
        created_since=None if created is None else created - _TASK_SESSION_SKEW,
        order_by=SessionOrder.CREATED_AT_ASC,
        # One more than shown says whether more were left out.
        limit=MAX_TASK_SESSIONS + 1,
    )


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        instant = datetime.fromisoformat(value)
    except ValueError:
        return None
    return instant if instant.tzinfo is not None else None


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
        return StoreTaskReader(
            store, source={"kind": "sqlite", "path": str(path)}, sessions=_session_store(target)
        )
    if target.postgres_dsn is None:
        raise AssertionError("Resolved Postgres target has no DSN.")
    from cayu.storage import PostgresTaskStore

    store = PostgresTaskStore(
        target.postgres_dsn, schema_mode=schema.SchemaMode.VALIDATE, read_only=True
    )
    return StoreTaskReader(store, source={"kind": "postgres"}, sessions=_session_store(target))


def _session_store(target: SessionStoreTarget) -> SessionStore:
    """The session store in the task store's database, opened read-only."""

    from cayu.storage import migrations as schema

    if target.backend is SessionStoreBackend.SQLITE:
        from cayu.storage import SQLiteSessionStore

        assert target.sqlite_path is not None
        return SQLiteSessionStore(
            target.sqlite_path, schema_mode=schema.SchemaMode.VALIDATE, read_only=True
        )
    from cayu.storage import PostgresSessionStore

    assert target.postgres_dsn is not None
    return PostgresSessionStore(
        target.postgres_dsn, schema_mode=schema.SchemaMode.VALIDATE, read_only=True
    )


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
        "error_code": task_error_code(task),
        "error_summary": task_error_summary(task),
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


def _type_query(types: tuple[str, ...]) -> dict[str, str | list[str] | None]:
    # One type goes as `type`, which every server reads; several as `types`.
    return {
        "type": types[0] if len(types) == 1 else None,
        "types": list(types) if len(types) > 1 else None,
    }


def _is_task_row(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    row = cast("dict[str, Any]", value)
    return all(isinstance(row.get(key), str) for key in ("id", "status", "type"))
