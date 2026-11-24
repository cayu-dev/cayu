"""Task control-plane actions preserve the task store's execution ownership."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.server._inbox_cursor import InboxCursorSeal
from cayu.server._task_index import read_task_index
from cayu.server.auth import AuthContext
from cayu.server.contracts import ApiTaskDetail, ApiTaskIndex
from cayu.tasks.bulk_cancellation import (
    MAX_BULK_CANCELLATION_TASKS,
    TaskBulkCancellation,
    cancel_tasks,
)
from cayu.tasks.index import TaskCountsCheck
from cayu.tasks.progress import TaskProgress
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import TaskStatus
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
#: The longest `POST /tasks/cancellations` waits for the tasks to settle.
TASK_BULK_CANCEL_MAX_WAIT_SECONDS = 60.0


TaskCancelReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=TASK_CANCEL_REASON_MAX_CHARS),
]


class TaskCancelBody(BaseModel):
    """Why the task is cancelled; recorded as its error (``task_cancelled``)."""

    model_config = ConfigDict(extra="forbid")

    reason: TaskCancelReason | None = None


class TaskBulkCancelBody(BaseModel):
    """Which tasks to cancel (as `GET /tasks` filters them), why, and how long to wait."""

    model_config = ConfigDict(extra="forbid")

    type: NonBlankString | None = None
    types: list[NonBlankString] | None = None
    status: TaskStatus | None = None
    session_id: NonBlankString | None = None
    parent_task_id: NonBlankString | None = None
    assigned_agent_name: NonBlankString | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    reason: TaskCancelReason | None = None
    #: How long the request waits for the tasks to settle; 0 returns once each is asked.
    wait_seconds: float = Field(default=0.0, ge=0, le=TASK_BULK_CANCEL_MAX_WAIT_SECONDS)
    max_tasks: int = Field(
        default=MAX_BULK_CANCELLATION_TASKS, ge=1, le=MAX_BULK_CANCELLATION_TASKS
    )


class ApiFencedTask(BaseModel):
    """A task still claimed after its worker stopped renewing its lease."""

    id: str
    type: str
    status: str
    worker_id: str | None
    lease_expires_at: str | None
    #: Whether Cayu Cloud's worker exit record shows the worker's run stopped; null when the
    #: worker id names no Cloud run or the record couldn't be read.
    worker_run_stopped: bool | None
    #: Whether ``POST /fenced-tasks/settle`` settles it now. A session-attached or
    #: retry-series task needs the step ``cayu inspect --fenced-tasks`` gives instead.
    settleable: bool = False


class ApiFencedTasks(BaseModel):
    tasks: list[ApiFencedTask]
    #: More fenced tasks exist than were listed.
    truncated: bool
    #: ``read``, ``unsupported`` (not on a Cayu Cloud that records worker exits) or
    #: ``unavailable`` (the read failed).
    worker_exits: str
    worker_exits_message: str | None = None


class ApiFencedTasksSettled(BaseModel):
    settled: list[ApiFencedTask]
    worker_exits: str
    worker_exits_message: str | None = None


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

    def _cancellation_error(reason: str | None) -> dict[str, Any] | None:
        if reason is None:
            return None
        return {"code": TASK_CANCELLED_ERROR_CODE, "message": cayu_app.redact_json(reason)}

    @router.post("/tasks/{task_id}/cancel", response_model=ApiTaskDetail, dependencies=protected)
    async def cancel_task(task_id: NonBlankString, body: TaskCancelBody | None = None):
        store = cayu_app.task_store
        if store is None:
            raise HTTPException(404, "Task store is not configured.")
        error = _cancellation_error(None if body is None else body.reason)
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

    @router.post(
        "/tasks/cancellations",
        response_model=TaskBulkCancellation,
        dependencies=protected,
        description=(
            "Cancel every unfinished task the filters match (at least one filter), as "
            "`POST /tasks/{task_id}/cancel` cancels one, and wait up to `wait_seconds` for "
            "them to settle. `has_more` says more matched than `max_tasks`; send it again."
        ),
    )
    async def cancel_matching_tasks(body: TaskBulkCancelBody):
        store = cayu_app.task_store
        if store is None:
            raise HTTPException(404, "Task store is not configured.")
        filters = body.model_dump(
            include={
                "type",
                "types",
                "status",
                "session_id",
                "parent_task_id",
                "assigned_agent_name",
                "labels",
            },
            exclude_defaults=True,
        )
        if not filters:
            raise HTTPException(422, "Give at least one filter; cancelling every task is refused.")
        try:
            query = TaskQuery.model_validate(filters)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_context=False)) from exc
        error = _cancellation_error(body.reason)
        return await cancel_tasks(
            store,
            query,
            error=error,
            wait_seconds=body.wait_seconds,
            max_tasks=body.max_tasks,
        )

    @router.get(
        "/fenced-tasks",
        response_model=ApiFencedTasks,
        dependencies=protected,
        description=(
            "Tasks fenced after their worker was lost, with whether Cayu Cloud recorded "
            "that worker's run stopped. `cayu task fenced` reads it."
        ),
    )
    async def fenced_tasks():
        from cayu.tasks.cloud_worker_exits import fenced_tasks_report

        store = cayu_app.task_store
        if store is None:
            raise HTTPException(501, "Task store is not configured.")
        return await fenced_tasks_report(store)

    @router.post(
        "/fenced-tasks/settle",
        response_model=ApiFencedTasksSettled,
        dependencies=protected,
        description=(
            "Settle the fenced tasks whose worker run Cayu Cloud recorded as stopped: "
            "restartable ones are requeued, the rest cancelled. "
            "`cayu task fenced --settle` calls it."
        ),
    )
    async def settle_fenced_tasks():
        from cayu.tasks.cloud_worker_exits import settle_fenced_tasks_report

        store = cayu_app.task_store
        if store is None:
            raise HTTPException(501, "Task store is not configured.")
        return await settle_fenced_tasks_report(store)

    @router.get(
        "/tasks/{task_id}/progress",
        response_model=TaskProgress,
        dependencies=protected,
        description=(
            "How far the task got, as its handler reported it to the app's task progress "
            "store: `done` of `total` units and `counts` by outcome. 404 when it reported "
            "nothing or the app keeps no progress store."
        ),
    )
    async def task_progress(task_id: NonBlankString):
        store = cayu_app.task_progress_store
        if store is None:
            raise HTTPException(404, "Task progress store is not configured.")
        if cayu_app.task_store is None or await cayu_app.task_store.load_task(task_id) is None:
            raise HTTPException(404, "Task was not found.")
        progress = await store.load_progress(task_id)
        if progress is None:
            raise HTTPException(404, "The task reported no progress.")
        return progress

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
