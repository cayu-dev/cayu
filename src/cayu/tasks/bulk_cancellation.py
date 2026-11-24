"""Cancel every task a query matches, and wait until they have settled.

``cancel_tasks(store, TaskQuery(labels={"job": "close"}), wait_seconds=30)`` cancels each
matching task that hasn't finished, as ``TaskStore.cancel_task`` does one: idle work is
cancelled at once, and a task a worker runs is asked to stop and is cancelled once its
handler has stopped. It then reads the asked ones again until each has finished or
``wait_seconds`` pass, and says which settled, which were still stopping and which the
store refused (a managed schedule, or a lifecycle that can't be cancelled).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from cayu.tasks.index import TaskIndexQuery, decode_task_index_cursor
from cayu.tasks.queries import TaskOrder, TaskQuery, copy_task_query
from cayu.tasks.records import _TERMINAL_TASK_STATUSES
from cayu.tasks.store import TaskStore

#: The most tasks one call cancels; the rest are left, and ``has_more`` says so.
MAX_BULK_CANCELLATION_TASKS = 1000
#: The longest one call waits for the asked tasks to settle.
MAX_BULK_CANCELLATION_WAIT_SECONDS = 3600.0
_PAGE = 500
_OFFSET_SCANS = 5
_CANCELLATION_REQUESTED = frozenset({"cancellation_requested", "retry_cancellation_requested"})


class TaskCancellationRefusal(BaseModel):
    """A matching task the store would not cancel, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    reason: str


class TaskBulkCancellation(BaseModel):
    """What ``cancel_tasks`` did to the tasks its query matched."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Tasks that have finished, whatever their end: cancelled, or completed or failed
    #: before the cancellation reached them.
    settled: tuple[str, ...] = ()
    #: Tasks asked to stop whose worker hadn't stopped them when the wait ended.
    stopping: tuple[str, ...] = ()
    refused: tuple[TaskCancellationRefusal, ...] = ()
    #: More tasks matched than one call cancels; call again for the rest.
    has_more: bool = False

    @property
    def done(self) -> bool:
        """Whether every matching task has finished."""
        return not self.stopping and not self.refused and not self.has_more


async def _matching_unfinished(
    store: TaskStore, query: TaskQuery, max_tasks: int
) -> tuple[list[str], bool]:
    # Read every id before cancelling any. The task index pages by creation time, which
    # no worker changes, so a task that leaves the query between pages (claimed,
    # finished) can't shift a later match out of reach, as an offset would.
    ids: list[str] = []
    seen: set[str] = set()

    def keep(tasks) -> bool:
        for task in tasks:
            if task.status not in _TERMINAL_TASK_STATUSES and task.id not in seen:
                if len(ids) == max_tasks:
                    return True
                ids.append(task.id)
                seen.add(task.id)
        return False

    filters = query.model_copy(
        update={"limit": _PAGE, "offset": 0, "order_by": TaskOrder.CREATED_AT_ASC}
    )
    after = None
    try:
        while True:
            page = await store.query_task_index(TaskIndexQuery(filters=filters, after=after))
            if keep(page.items):
                return ids, True
            if page.next_position is None:
                return ids, False
            after = decode_task_index_cursor(page.next_position)
    except NotImplementedError:
        pass
    # A store without a task index pages by offset, which a task leaving the query between
    # pages shifts. Scan again until a scan finds nothing new, so a match an earlier scan
    # stepped over is still found.
    for _scan in range(_OFFSET_SCANS):
        found = len(ids)
        offset = 0
        while True:
            tasks = await store.list_tasks(filters.model_copy(update={"offset": offset}))
            if keep(tasks):
                return ids, True
            if len(tasks) < _PAGE:
                break
            offset += _PAGE
        if len(ids) == found:
            break
    return ids, False


async def cancel_tasks(
    store: TaskStore,
    query: TaskQuery,
    *,
    error: dict[str, Any] | None = None,
    wait_seconds: float = 0.0,
    poll_seconds: float = 0.5,
    max_tasks: int = MAX_BULK_CANCELLATION_TASKS,
) -> TaskBulkCancellation:
    """Cancel each unfinished task ``query`` matches and wait up to ``wait_seconds``.

    ``query``'s ``limit``, ``offset`` and ``order_by`` are ignored: every match is read,
    oldest first, up to ``max_tasks``. ``error`` is recorded on each as
    ``cancel_task``'s is.
    """

    query = copy_task_query(query)
    if query.fenced_after_worker_loss is not None:
        # The task index can't serve it, and offsets over a filter workers keep changing
        # would skip matches.
        raise ValueError("cancel_tasks does not take fenced_after_worker_loss.")
    if not 1 <= max_tasks <= MAX_BULK_CANCELLATION_TASKS:
        raise ValueError(f"max_tasks must be 1 to {MAX_BULK_CANCELLATION_TASKS}.")
    if not 0 <= wait_seconds <= MAX_BULK_CANCELLATION_WAIT_SECONDS:
        raise ValueError(f"wait_seconds must be 0 to {MAX_BULK_CANCELLATION_WAIT_SECONDS:g}.")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    deadline = time.monotonic() + wait_seconds
    from cayu.tasks.scheduling import TaskScheduleConflict

    task_ids, has_more = await _matching_unfinished(store, query, max_tasks)
    settled: list[str] = []
    stopping: list[str] = []
    refused: list[TaskCancellationRefusal] = []
    for task_id in task_ids:
        try:
            task = await store.cancel_task(task_id, error)
        except KeyError:
            # Deleted since it was listed: nothing is left to stop.
            settled.append(task_id)
            continue
        except TaskScheduleConflict:
            refused.append(
                TaskCancellationRefusal(
                    task_id=task_id,
                    reason="Managed schedules require revision-fenced cancellation.",
                )
            )
            continue
        except ValueError:
            # Finished since it was listed (a prerequisite's cancellation skipped it, or it
            # completed), or already asked to stop: only a held lifecycle is a refusal.
            current = await store.load_task(task_id)
            if current is None or current.status in _TERMINAL_TASK_STATUSES:
                settled.append(task_id)
            elif current.status_reason in _CANCELLATION_REQUESTED:
                stopping.append(task_id)
            else:
                refused.append(
                    TaskCancellationRefusal(
                        task_id=task_id,
                        reason="Task cancellation conflicts with its current lifecycle.",
                    )
                )
            continue
        (settled if task.status in _TERMINAL_TASK_STATUSES else stopping).append(task_id)
    while stopping and time.monotonic() < deadline:
        await asyncio.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
        still: list[str] = []
        for task_id in stopping:
            task = await store.load_task(task_id)
            if task is None or task.status in _TERMINAL_TASK_STATUSES:
                settled.append(task_id)
            else:
                still.append(task_id)
        stopping = still
    return TaskBulkCancellation(
        settled=tuple(settled),
        stopping=tuple(stopping),
        refused=tuple(refused),
        has_more=has_more,
    )


__all__ = [
    "MAX_BULK_CANCELLATION_TASKS",
    "MAX_BULK_CANCELLATION_WAIT_SECONDS",
    "TaskBulkCancellation",
    "TaskCancellationRefusal",
    "cancel_tasks",
]
