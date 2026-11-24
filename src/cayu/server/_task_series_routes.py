"""Recurring task series kept in the task store: list, show, pause, resume and delete.

`cayu task series ... --application NAME` reads and changes them through these routes.
A series spans every task it queues, so a caller whose task reads are scoped to some
labels is refused (403), as for the task counts check.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, StringConstraints

from cayu.tasks.series import TaskSeriesState

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.tasks.store import TaskStore

SeriesId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ApiTaskSeries(BaseModel):
    """One stored series. `state` is the whole `TaskSeriesState` as JSON."""

    model_config = ConfigDict(extra="forbid")

    series_id: str
    paused: bool
    revision: int
    next_task_id: str | None
    updated_at: datetime
    state: dict[str, Any]

    @classmethod
    def of(cls, state: TaskSeriesState, redact: Callable[[Any], Any]) -> ApiTaskSeries:
        """``state`` with every string value redacted as task details are; keys stay."""

        return cls(
            series_id=redact(state.series.series_id),
            paused=state.paused,
            revision=state.revision,
            next_task_id=None if state.next_task_id is None else redact(state.next_task_id),
            updated_at=state.updated_at,
            state=_redact_values(state.model_dump(mode="json"), redact),
        )


def _redact_values(value: Any, redact: Callable[[Any], Any]) -> Any:
    if type(value) is str:
        return redact(value)
    if type(value) is list:
        return [_redact_values(item, redact) for item in value]
    if type(value) is dict:
        return {key: _redact_values(item, redact) for key, item in value.items()}
    return value


class ApiTaskSeriesPage(BaseModel):
    """Stored series in `series_id` order."""

    model_config = ConfigDict(extra="forbid")

    series: list[ApiTaskSeries]


class ApiTaskSeriesDeleted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str
    deleted: bool


def register_task_series_routes(
    router: APIRouter, *, cayu_app: CayuApp, protected: list[Depends]
) -> None:
    async def store() -> TaskStore:
        from cayu.resource_access import current_data_bounds

        task_store = cayu_app.task_store
        if task_store is None:
            raise HTTPException(501, "Task store is not configured.")
        if not task_store.supports_task_series:
            raise HTTPException(501, "This task store keeps no task series.")
        if await current_data_bounds("tasks") is not None:
            raise HTTPException(403, "Task series cover every task they queue.")
        return task_store

    @router.get(
        "/task-series",
        response_model=ApiTaskSeriesPage,
        dependencies=protected,
        description="Stored recurring task series. `cayu task series list` reads it.",
    )
    async def list_task_series(
        limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0)
    ):
        task_store = await store()
        states = await task_store.list_task_series(limit=limit, offset=offset)
        return ApiTaskSeriesPage(
            series=[ApiTaskSeries.of(state, cayu_app.redact_json) for state in states]
        )

    @router.get(
        "/task-series/{series_id:path}",
        response_model=ApiTaskSeries,
        dependencies=protected,
        description="One stored series; 404 when there is none.",
    )
    async def show_task_series(series_id: SeriesId):
        task_store = await store()
        state = await task_store.load_task_series(series_id)
        if state is None:
            raise HTTPException(404, "Task series was not found.")
        return ApiTaskSeries.of(state, cayu_app.redact_json)

    async def set_paused(series_id: str, paused: bool) -> ApiTaskSeries:
        task_store = await store()
        try:
            return ApiTaskSeries.of(
                await task_store.set_task_series_paused(series_id, paused), cayu_app.redact_json
            )
        except KeyError:
            raise HTTPException(404, "Task series was not found.") from None

    @router.post(
        "/task-series/{series_id:path}/pause",
        response_model=ApiTaskSeries,
        dependencies=protected,
        description="Pause a series: an occurrence that ends queues nothing more.",
    )
    async def pause_task_series(series_id: SeriesId):
        return await set_paused(series_id, True)

    @router.post(
        "/task-series/{series_id:path}/resume",
        response_model=ApiTaskSeries,
        dependencies=protected,
        description="Resume a series; queue its next occurrence if nothing is queued.",
    )
    async def resume_task_series(series_id: SeriesId):
        return await set_paused(series_id, False)

    @router.post(
        "/task-series/{series_id:path}/delete",
        response_model=ApiTaskSeriesDeleted,
        dependencies=protected,
        description="Delete a series; its tasks stay. 404 when there is none.",
    )
    async def delete_task_series(series_id: SeriesId):
        task_store = await store()
        if not await task_store.delete_task_series(series_id):
            raise HTTPException(404, "Task series was not found.")
        return ApiTaskSeriesDeleted(series_id=series_id, deleted=True)
