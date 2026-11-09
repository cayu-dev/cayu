"""Hand a caught cleanup failure's unfinished teardown to the owner that waits for it.

A runner, binding or factory release can fail while its teardown continues in an
authenticated cleanup task carried by the failure. A handler that records such a
failure as a diagnostic, instead of raising it, must pass it through
``keep_cleanup_owners`` first. Otherwise the task leaves every retained-cleanup
account and drains report quiescence while it still runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable, Sequence
from typing import TypeVar

from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    combine_environment_factory_cleanup_settlement_tasks,
    environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_tasks,
)

# How long a retry waits for teardown an earlier failed cleanup handed off.
KEPT_TEARDOWN_TIMEOUT_S = 15.0

_Source = TypeVar("_Source", bound="BaseException | Sequence[asyncio.Task[None]]")


def keep_cleanup_owners(
    source: _Source,
    *,
    into: BaseException | list[asyncio.Task[None]],
) -> _Source:
    """Move every cleanup owner ``source`` carries into ``into`` and return ``source``.

    ``into`` is either a list the caller later hands on, or the failure that
    carries cleanup ownership to the environment lifecycle; that failure keeps
    the owner it already carries alongside the new ones.
    """

    tasks = (
        environment_factory_cleanup_settlement_tasks(source)
        if isinstance(source, BaseException)
        else tuple(source)
    )
    if isinstance(into, list):
        into.extend(task for task in tasks if task not in into)
        return source
    # An owner already in ``into``'s tree (``into`` groups ``source``) is found
    # there; carrying it again at the top would settle it twice.
    known = set(environment_factory_cleanup_settlement_tasks(into))
    tasks = tuple(task for task in tasks if task not in known)
    if not tasks:
        return source
    current = environment_factory_cleanup_settlement_task(into)
    owner = combine_environment_factory_cleanup_settlement_tasks(
        (*(() if current is None else (current,)), *tasks),
        task_name="cayu-kept-environment-cleanup",
        failure_message="Kept environment cleanup failed to settle.",
    )
    assert owner is not None
    attach_environment_factory_cleanup_settlement_task(into, owner)
    return source


class KeptTeardownStillRunning(TimeoutError):
    """An earlier cleanup's handed-off teardown outlived the bound; retry later."""


class KeptTeardowns:
    """Teardown that failed best-effort cleanups handed off, kept per resource.

    A handler passes ``bucket(key)`` to ``keep_cleanup_owners``; every later
    attempt to use or dispose of that resource calls ``await_settled(key)`` first.
    """

    def __init__(self) -> None:
        self._tasks: dict[Hashable, list[asyncio.Task[None]]] = {}

    def bucket(self, key: Hashable) -> list[asyncio.Task[None]]:
        # Buckets a handler opened without keeping anything are dropped here.
        for empty in [other for other, tasks in self._tasks.items() if not tasks]:
            del self._tasks[empty]
        return self._tasks.setdefault(key, [])

    async def await_settled(self, key: Hashable) -> None:
        """Wait for ``key``'s kept teardown, which stays kept until it has settled.

        Past the bound it raises ``KeptTeardownStillRunning`` and keeps it.
        """

        tasks = tuple(self._tasks.get(key, ()))
        if tasks:
            await asyncio.wait(tasks, timeout=KEPT_TEARDOWN_TIMEOUT_S)
        remaining = [task for task in self._tasks.pop(key, ()) if not task.done()]
        for task in tasks:
            # Its failure was already reported where it was handed off.
            if task.done() and not task.cancelled():
                task.exception()
        if remaining:
            self._tasks[key] = remaining
            raise KeptTeardownStillRunning(
                "An earlier cleanup's handed-off teardown is still running; retry later."
            )
