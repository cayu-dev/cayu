"""Translate content-free store observations for the shared durable worker timer."""

from __future__ import annotations

from time import monotonic

from cayu._validation import revalidate_model_input
from cayu.tasks.base import TaskQuery, TaskStore
from cayu.tasks.scheduling import TaskScheduleWakeup


async def next_schedule_wake_at(
    store: TaskStore, queries: tuple[TaskQuery | None, ...]
) -> float | None:
    if not store.supports_task_scheduling:
        return None
    deadline: float | None = None
    for query in queries:
        observation = revalidate_model_input(
            await store.next_task_schedule_wakeup(query), TaskScheduleWakeup
        )
        # The store samples as_of during the awaited read, not before it.
        # Anchoring to request start can schedule an early claim under latency;
        # the following observation may then miss a deadline crossed in flight.
        # Response time is conservative; the ordinary poll remains an upper bound.
        observed_at = monotonic()
        for timestamp in (observation.next_available_at, observation.next_expiry_at):
            if timestamp is None:
                continue
            candidate = observed_at + max(0.0, (timestamp - observation.as_of).total_seconds())
            deadline = candidate if deadline is None else min(deadline, candidate)
    # Overdue maintenance keeps the worker's existing bounded poll cadence.
    # A hint must not create a spin loop while another worker owns the row.
    return deadline
