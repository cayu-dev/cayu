"""In-memory task index snapshot, retaining the existing store lock."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from cayu.resource_access import current_data_bounds
from cayu.tasks.access import visible
from cayu.tasks.index import (
    TASK_SEARCH_FIELDS,
    TaskIndexAccumulator,
    TaskIndexPredicate,
    TaskIndexQuery,
    TaskIndexResult,
    TaskSearchMatch,
)
from cayu.tasks.queries import _sort_tasks, _task_matches

if TYPE_CHECKING:
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.records import Task


def _native_match(query: TaskIndexQuery, search: TaskSearchMatch | None) -> TaskIndexPredicate:
    """Stored-value matching for a read without a caller predicate."""

    needle = None if query.filters.q is None else query.filters.q.casefold()

    def matches(task: Task) -> bool:
        if needle is None:
            return True
        fields = tuple(
            None if (value := getattr(task, name)) is None else str(value)
            for name in TASK_SEARCH_FIELDS
        )
        if search is not None:
            return search(fields)
        return any(value is not None and needle in value.casefold() for value in fields)

    return matches


async def query_index(
    store: InMemoryTaskStore,
    query: TaskIndexQuery,
    *,
    matches: TaskIndexPredicate | None = None,
    search: TaskSearchMatch | None = None,
) -> TaskIndexResult:
    query = TaskIndexQuery.model_validate(query.model_dump(mode="python", warnings=False))
    bounds = await current_data_bounds("tasks")
    filters = query.filters.model_copy(update={"q": None, "status": None})
    result = TaskIndexAccumulator(query, matches or _native_match(query, search))
    async with store._lock:
        as_of = store._clock()
        for index, task in enumerate(_sort_tasks(list(store._tasks.values()), filters.order_by)):
            if _task_matches(task, filters) and (bounds is None or visible(task, bounds)):
                result.visit(task.model_copy(deep=True))
            if index % 256 == 0:
                await asyncio.sleep(0)
        return result.result(as_of)
