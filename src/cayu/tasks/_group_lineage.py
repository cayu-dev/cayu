"""Authenticate automatic retry descendants without changing graph membership."""

from __future__ import annotations

from collections.abc import Mapping

from cayu.tasks.base import Task, _task_retry_successor_id, copy_task
from cayu.tasks.groups import TaskGroupSnapshot, TaskGroupUnavailable


def validate_lineage(
    snapshot: TaskGroupSnapshot,
    tasks: Mapping[str, Task],
    roots: Mapping[str, str],
) -> None:
    """Check the store-owned index against both ends of every exact retry link.

    A caller-authored series ID is not membership. Only automatic settlement
    publishes this index, in the same transaction as source and successor.
    """
    if len(roots) > 128 * 99:
        raise TaskGroupUnavailable("Group retry lineage exceeds its admission bound.")
    for identity, root_id in roots.items():
        if (
            root_id not in snapshot.receipt.member_task_ids
            or identity in snapshot.receipt.graph.task_ids
        ):
            raise TaskGroupUnavailable("Retry lineage contradicts group membership.")
        task = tasks.get(identity)
        root = tasks.get(root_id)
        if task is None or root is None:
            raise TaskGroupUnavailable("Retry lineage lost retained task evidence.")
        task = copy_task(task)
        series = task.retry_series
        root_series = root.retry_series
        if series is None or root_series is None or series.predecessor_task_id is None:
            raise TaskGroupUnavailable("Retry lineage has no predecessor authority.")
        predecessor = tasks.get(series.predecessor_task_id)
        previous = None if predecessor is None else predecessor.retry_series
        if (
            previous is None
            or previous.successor_task_id != identity
            or series.attempt != previous.attempt + 1
            or series.series_id != root_series.series_id
            or previous.series_id != series.series_id
            or series.causal_budget_id != root_series.causal_budget_id
            or series.policy != root_series.policy
            or series.started_at != root_series.started_at
            or identity != _task_retry_successor_id(series.series_id, series.attempt)
            or (
                series.predecessor_task_id != root_id
                and roots.get(series.predecessor_task_id) != root_id
            )
        ):
            raise TaskGroupUnavailable("Retry lineage contradicts exact settlement authority.")
    for identity in (*snapshot.receipt.member_task_ids, *roots):
        series = tasks[identity].retry_series
        if series is not None and series.successor_task_id is not None:
            expected_root = roots.get(identity, identity)
            if roots.get(series.successor_task_id) != expected_root:
                raise TaskGroupUnavailable("Retry successor is missing its group ownership index.")
