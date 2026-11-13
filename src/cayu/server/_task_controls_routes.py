"""Task control-plane actions preserve the task store's execution ownership."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, StringConstraints

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.server._inbox_cursor import InboxCursorSeal
from cayu.server._task_index import read_task_index
from cayu.server.auth import AuthContext
from cayu.server.contracts import ApiTaskDetail, ApiTaskIndex
from cayu.tasks.index import TaskCountsCheck
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.scheduling import TaskScheduleConflict

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.tasks.records import Task


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
#: The longest cancellation reason the task records as its error message.
TASK_CANCEL_REASON_MAX_CHARS = 4096
#: The error code of a task cancelled through the API with a reason.
TASK_CANCELLED_ERROR_CODE = "task_cancelled"


class TaskCancelBody(BaseModel):
    """Why the task is cancelled; recorded as its error (``task_cancelled``)."""

    model_config = ConfigDict(extra="forbid")

    reason: (
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True, min_length=1, max_length=TASK_CANCEL_REASON_MAX_CHARS
            ),
        ]
        | None
    ) = None


def register_task_controls_routes(
    router: APIRouter,
    *,
    cayu_app: CayuApp,
    protected: list[Depends],
    serialize_task: Callable[[Any, Task], dict[str, Any]],
    serialize_list: Callable[[Any, Task], dict[str, Any]],
    cursor_seal: InboxCursorSeal,
    optional_auth_context: Depends,
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

    @router.post("/tasks/{task_id}/cancel", response_model=ApiTaskDetail, dependencies=protected)
    async def cancel_task(task_id: NonBlankString, body: TaskCancelBody | None = None):
        store = cayu_app.task_store
        if store is None:
            raise HTTPException(404, "Task store is not configured.")
        reason = None if body is None else body.reason
        error = (
            None
            if reason is None
            else {"code": TASK_CANCELLED_ERROR_CODE, "message": cayu_app.redact_json(reason)}
        )
        try:
            task = await store.cancel_task(task_id, error)
        except KeyError:
            raise HTTPException(404, "Task was not found.") from None
        except TaskScheduleConflict:
            raise HTTPException(
                409, "Managed schedules require revision-fenced cancellation."
            ) from None
        except ValueError:
            raise HTTPException(
                409, "Task cancellation conflicts with its current lifecycle."
            ) from None
        return serialize_task(cayu_app, task)

    @router.get(
        "/tasks/counts/check",
        response_model=TaskCountsCheck,
        dependencies=protected,
        description=(
            "Compare the stored task counts that `include=counts` serves with a recount "
            "of every task. `cayu task counts --recompute` reads it."
        ),
    )
    async def check_task_counts():
        store = cayu_app.task_store
        if store is None:
            raise HTTPException(501, "Task store is not configured.")
        try:
            return await store.check_task_counts()
        except NotImplementedError:
            raise HTTPException(501, "This task store keeps no task counts.") from None
        except ResourceAccessDenied:
            raise HTTPException(403, "Task counts cover every task.") from None

    @router.get("/tasks/{task_id}/dependents", response_model=ApiTaskIndex, dependencies=protected)
    async def dependents(
        task_id: NonBlankString,
        auth_context: AuthContext | None = auth_default,
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=1000),
    ):
        store = cayu_app.task_store
        if store is None or await store.load_task(task_id) is None:
            raise HTTPException(404, "Task was not found.")
        return await read_task_index(
            cayu_app,
            TaskQuery(limit=limit, order_by=TaskOrder.CREATED_AT_ASC),
            include_internal=True,
            cursor=cursor,
            prerequisite_task_id=task_id,
            serialize=serialize_list,
            seal=cursor_seal,
            actor=auth_context,
        )
