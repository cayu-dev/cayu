"""``GET /api/tasks/latest``: the newest finished tasks, or the newest per input key."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import HTTPException, Query, Response
from pydantic import ValidationError

from cayu.server.contracts import ApiTaskDetail
from cayu.tasks.queries import TaskQuery, latest_tasks_key_path, task_input_key
from cayu.tasks.records import TaskStatus

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.tasks.records import Task
    from cayu.tasks.store import TaskStore

# How many groups one store read asks for while paging over the shown groups, and how
# many such reads one request may make: each runs the store's grouping query again.
_GROUP_READ_SIZE = 1000
_MAX_GROUP_READS = 10
_TOO_DEEP = (
    f"With by, a request reads at most {_GROUP_READ_SIZE * _MAX_GROUP_READS} keys. "
    "Narrow the tasks with type, status or session_id, or use a smaller offset."
)


class _PageTooDeep(Exception):
    """The page lies past the keys one request may read."""


async def _shown_groups(
    store: TaskStore, query: TaskQuery, by: str, redact: Callable[[Any], Any]
) -> list[Task]:
    """The page of ``query`` over the groups whose key the response shows unmasked.

    A key that redaction masks forms no group: one row per hidden value would count the
    hidden values and tell which tasks share one. The page is cut after those groups
    are left out, so it has no gaps that would count them either. A page that is not
    full after the reads one request may make is refused, not returned short.
    """

    path = latest_tasks_key_path(by)
    page: list[Task] = []
    skipped = read = 0
    for _ in range(_MAX_GROUP_READS):
        groups = await store.latest_tasks(
            query.model_copy(update={"offset": read, "limit": _GROUP_READ_SIZE}), by=by
        )
        for task in groups:
            key = task_input_key(task.input, path)
            if redact(key) != key:
                continue
            if skipped < query.offset:
                skipped += 1
            elif len(page) < query.limit:
                page.append(task)
        read += len(groups)
        if len(groups) < _GROUP_READ_SIZE or len(page) == query.limit:
            return page
    raise _PageTooDeep


def register_task_latest_route(
    router: APIRouter,
    *,
    cayu_app: CayuApp,
    protected: list[Depends],
    serialize_task: Callable[[Any, Task], dict[str, Any]],
) -> None:
    """Register before ``/tasks/{task_id}``, which would otherwise take ``latest``."""

    @router.get("/tasks/latest", response_model=list[ApiTaskDetail], dependencies=protected)
    async def latest_tasks(
        response: Response,
        task_type: str | None = Query(default=None, alias="type"),
        types: Annotated[list[str] | None, Query()] = None,
        status: TaskStatus | None = None,
        by: str | None = Query(
            default=None,
            description=(
                "A field of the task's input, such as input.repository: the newest task "
                "for each distinct string value of it."
            ),
        ),
        session_id: str | None = None,
        parent_task_id: str | None = None,
        assigned_agent_name: str | None = None,
        limit: int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        """The newest finished tasks by completion time, newest first, with their results."""

        response.headers["Cache-Control"] = "private, no-store"
        store = cayu_app.task_store
        if store is None:
            raise HTTPException(404, "Task store is not configured.")
        try:
            query = TaskQuery(
                type=task_type,
                types=None if not types else tuple(types),
                status=status,
                session_id=session_id,
                parent_task_id=parent_task_id,
                assigned_agent_name=assigned_agent_name,
                limit=limit,
                offset=offset,
            )
            if by is not None:
                latest_tasks_key_path(by)
        except ValidationError as exc:
            raise HTTPException(
                422, exc.errors(include_url=False, include_input=False, include_context=False)
            ) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        # Selectors match the public projection, as on `GET /api/tasks`: a value that
        # redaction would mask never matches, so the route can't confirm a hidden value.
        selectors = {
            "session_id": session_id,
            "parent_task_id": parent_task_id,
            "assigned_agent_name": assigned_agent_name,
        }
        requested = [*(query.types or ()), *(value for value in selectors.values() if value)]
        if query.type is not None:
            requested.append(query.type)
        if any(cayu_app.redact_json(value) != value for value in requested):
            return []
        try:
            if by is None:
                tasks = await store.latest_tasks(query)
            else:
                tasks = await _shown_groups(store, query, by, cayu_app.redact_json)
        except NotImplementedError:
            raise HTTPException(501, "This task store can't read the latest tasks.") from None
        except _PageTooDeep:
            raise HTTPException(422, _TOO_DEEP) from None
        rows = [serialize_task(cayu_app, task) for task in tasks]
        allowed_types = query.types or ((query.type,) if query.type is not None else None)
        return [
            row
            for row in rows
            if (allowed_types is None or row.get("type") in allowed_types)
            and all(value is None or row.get(field) == value for field, value in selectors.items())
        ]
