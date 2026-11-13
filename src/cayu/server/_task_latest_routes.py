"""``GET /api/tasks/latest``: the newest finished tasks, or the newest per input key."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import HTTPException, Query, Response
from pydantic import ValidationError

from cayu.server.contracts import ApiTaskDetail
from cayu.tasks.queries import TaskQuery, latest_tasks_key_path
from cayu.tasks.records import TaskStatus

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.tasks.records import Task


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
            tasks = await store.latest_tasks(query, by=by)
        except NotImplementedError:
            raise HTTPException(501, "This task store can't read the latest tasks.") from None
        rows = [serialize_task(cayu_app, task) for task in tasks]
        allowed_types = query.types or ((query.type,) if query.type is not None else None)
        return [
            row
            for row in rows
            if (allowed_types is None or row.get("type") in allowed_types)
            and all(value is None or row.get(field) == value for field, value in selectors.items())
        ]
