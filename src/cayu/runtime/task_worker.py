"""A generic durable task-worker loop for starting fresh sessions from tasks.

``TaskStoreDispatcher.run_worker`` resumes existing sessions from dispatch
requests. This helper covers the complementary shape used by, for example, the
PR-reviewer recipe: a worker that claims arbitrary :class:`Task`\\ s and starts a
*new* session for each. It owns the claim -> heartbeat -> handle -> loop cycle
plus optional expired-lease reclaim, so a caller only supplies a handler that
turns a claimed task into an ``app.run(...)``.

The handler owns the task's terminal state: run with ``RunRequest(task_id=...,
task_worker_id=...)`` so the runtime completes/fails the task, or call
``task_store.complete_task``/``fail_task`` explicitly. If the handler raises or
returns while the task is still active, the worker marks the task failed and keeps
going -- one bad task does not kill it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from cayu.runtime.tasks import Task, TaskQuery, TaskStatus, TaskStore

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp

TaskHandler = Callable[["CayuApp", Task, str], Awaitable[None]]


async def run_task_worker(
    app: CayuApp,
    task_store: TaskStore,
    handler: TaskHandler,
    *,
    worker_id: str,
    query: TaskQuery | None = None,
    lease_seconds: int = 300,
    poll_interval_s: float = 1.0,
    reclaim: bool = True,
    stop: asyncio.Event | None = None,
    max_tasks: int | None = None,
) -> int:
    """Claim and handle durable tasks until stopped; return the number handled.

    For each claimed task, ``handler(app, task, worker_id)`` is awaited while the
    task lease is heartbeated in the background. The handler typically builds a
    ``RunRequest(task_id=task.id, task_worker_id=worker_id, ...)`` and awaits
    ``app.run(...)`` so the runtime completes/fails the task.

    - ``query`` scopes which tasks this worker claims (e.g. by type / assigned agent).
    - ``lease_seconds`` is the claim lease; the lease is re-extended at ~1/3 of it.
    - ``poll_interval_s`` is how long to wait when no task is available.
    - ``reclaim`` reclaims expired leases (from dead workers) before each claim.
    - ``stop`` is an ``asyncio.Event`` for graceful shutdown.
    - ``max_tasks`` bounds the loop (useful for tests and one-shot drains).
    """
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive.")
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative.")

    handled = 0
    while (max_tasks is None or handled < max_tasks) and not _is_stopped(stop):
        if reclaim:
            await task_store.reclaim_expired(query=query)
        task = await task_store.claim_task(worker_id, query, lease_seconds=lease_seconds)
        if task is None:
            if await _wait_or_stop(poll_interval_s, stop):
                break
            continue
        await _handle_with_heartbeat(app, task_store, task, handler, worker_id, lease_seconds)
        handled += 1
    return handled


async def _handle_with_heartbeat(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
) -> None:
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_until(task_store, task.id, worker_id, lease_seconds, stop_heartbeat)
    )
    handler_error: Exception | None = None
    try:
        await handler(app, task, worker_id)
    except Exception as exc:  # a single bad task must not stop the worker
        handler_error = exc
    finally:
        stop_heartbeat.set()
        await heartbeat_task
    if handler_error is not None:
        await _safe_fail(task_store, task.id, worker_id, handler_error)
    else:
        await _safe_fail_unfinished(task_store, task.id, worker_id)


async def _heartbeat_until(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    stop: asyncio.Event,
) -> None:
    interval = max(1.0, lease_seconds / 3)
    while not stop.is_set():
        if await _wait_or_stop(interval, stop):
            return
        try:
            await task_store.heartbeat(task_id, worker_id, extend_seconds=lease_seconds)
        except Exception:
            # Task already terminal (handler finished) or lease lost -> stop beating.
            return


async def _safe_fail(task_store: TaskStore, task_id: str, worker_id: str, exc: Exception) -> None:
    # Task may already be terminal (e.g. the handler completed it); if the fail
    # is rejected, leave the task for lease reclaim.
    with contextlib.suppress(Exception):
        await task_store.fail_task(
            task_id,
            {"error": type(exc).__name__, "message": str(exc)[:500]},
            worker_id=worker_id,
        )


async def _safe_fail_unfinished(task_store: TaskStore, task_id: str, worker_id: str) -> None:
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return
    await _safe_fail(
        task_store,
        task_id,
        worker_id,
        RuntimeError("Task handler returned without completing or failing the task."),
    )


def _is_stopped(stop: asyncio.Event | None) -> bool:
    return stop is not None and stop.is_set()


async def _wait_or_stop(seconds: float, stop: asyncio.Event | None) -> bool:
    """Sleep for ``seconds`` or until ``stop`` is set. Returns True if stopped."""
    if stop is None:
        await asyncio.sleep(seconds)
        return False
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except TimeoutError:
        return False
