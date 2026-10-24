"""In-memory task index snapshot, retaining the existing store lock."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from cayu.resource_access import current_data_bounds
from cayu.tasks.access import visible
from cayu.tasks.index import (
    TaskIndexAccumulator,
    TaskIndexPredicate,
    TaskIndexQuery,
    TaskIndexResult,
)
from cayu.tasks.queries import _sort_tasks, _task_matches

if TYPE_CHECKING:
    from cayu.tasks.memory import InMemoryTaskStore


async def query_index(
    store: InMemoryTaskStore, query: TaskIndexQuery, *, matches: TaskIndexPredicate
) -> TaskIndexResult:
    query = TaskIndexQuery.model_validate(query.model_dump(mode="python", warnings=False))
    bounds = await current_data_bounds("tasks")
    filters = query.filters.model_copy(update={"q": None, "status": None})
    result = TaskIndexAccumulator(query, matches)
    async with store._lock:
        as_of = store._clock()
        for index, task in enumerate(_sort_tasks(list(store._tasks.values()), filters.order_by)):
            if _task_matches(task, filters) and (bounds is None or visible(task, bounds)):
                result.visit(task.model_copy(deep=True))
            if index % 256 == 0:
                await asyncio.sleep(0)
        return result.result(as_of)
