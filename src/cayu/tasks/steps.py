"""Saved step results for task handlers, keyed by task id and step name.

A task handler that does more than one piece of work can save each step's
result as it finishes. When the same task runs again, for example after a lost
worker or a restart, :meth:`TaskSteps.run` returns the saved result instead of
calling the step again, so the handler pays only for steps that never finished.

The parts stand alone. :class:`TaskStepStore` is the storage contract,
:class:`InMemoryTaskStepStore` is a process-local store for tests, and the
SQLite and PostgreSQL stores in :mod:`cayu.storage` keep results in the same
database as the task store. :class:`TaskSteps` is a small helper bound to one
task id. Nothing in the task worker requires any of them.

The first saved result for a step wins: a later save of the same step returns
the stored record unchanged, so two runs that race on one step agree on a
single result. Results are JSON values of at most
:data:`TASK_STEP_RESULT_MAX_BYTES` compact UTF-8 bytes. Store a reference (an
artifact id, a path, a commit sha) for anything larger.

A saved result says the step finished; it does not undo or repeat the step's
external effects. A step that changes another system should still use a stable
idempotency key there, because the worker can be lost after the effect and
before the save.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from cayu._validation import (
    DURABLE_METADATA_LIMITS,
    canonical_bounded_durable_json_bytes,
    require_durable_clean_nonblank,
)

#: Largest saved step result, in compact UTF-8 JSON bytes.
TASK_STEP_RESULT_MAX_BYTES = 256 * 1024
#: Largest task id or step name, in UTF-8 bytes.
TASK_STEP_KEY_MAX_BYTES = 512
#: Largest page :meth:`TaskStepStore.list_steps` returns.
TASK_STEP_LIST_MAX = 1000


class TaskStepRecord(BaseModel):
    """One saved step result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    step: str
    result: Any
    saved_at: datetime


def task_step_key(value: object, field_name: str) -> str:
    """Validate a task id or step name the way every store does."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TASK_STEP_KEY_MAX_BYTES:
        raise ValueError(f"{field_name} must not exceed {TASK_STEP_KEY_MAX_BYTES} UTF-8 bytes.")
    return value


def task_step_ids(task_id: object, step: object) -> tuple[str, str]:
    """Validate one ``(task_id, step)`` key."""

    return task_step_key(task_id, "task_id"), task_step_key(step, "step")


def optional_task_step_key(value: object, field_name: str) -> str | None:
    """Validate an optional step name, such as a list cursor or a single-step delete."""

    return None if value is None else task_step_key(value, field_name)


def encode_task_step_result(result: Any) -> str:
    """Return the canonical stored JSON text for ``result``, refusing oversized values."""

    return canonical_bounded_durable_json_bytes(
        result,
        "task step result",
        max_bytes=TASK_STEP_RESULT_MAX_BYTES,
        max_nodes=DURABLE_METADATA_LIMITS.max_nodes,
    ).decode("utf-8")


def task_step_list_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= TASK_STEP_LIST_MAX:
        raise ValueError(f"limit must be an integer between 1 and {TASK_STEP_LIST_MAX}.")
    return value


def task_step_saved_at_ms() -> int:
    """Store-side save time in UTC milliseconds, the precision every backend keeps."""

    return int(datetime.now(UTC).timestamp() * 1000)


def task_step_record(task_id: str, step: str, result_json: str, saved_at_ms: int) -> TaskStepRecord:
    """Rebuild a record from its stored columns; every read returns a fresh copy."""

    return TaskStepRecord(
        task_id=task_id,
        step=step,
        result=json.loads(result_json),
        saved_at=datetime.fromtimestamp(saved_at_ms / 1000, tz=UTC),
    )


class TaskStepStore(ABC):
    """Storage contract for saved task step results.

    Implementations key records by ``(task_id, step)``, keep the first saved
    result for a key, and validate keys and results with
    :func:`task_step_key` and :func:`encode_task_step_result` so every backend
    accepts and returns the same values. The store does not check that the task
    exists or who owns it; callers pass the id of the task they are running.
    """

    @abstractmethod
    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        """Return the saved result for one step, or ``None`` when there is none."""

    @abstractmethod
    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        """Save ``result`` unless the step already has one, and return the stored record."""

    @abstractmethod
    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        """Return one task's saved steps ordered by step name, starting after ``after``."""

    @abstractmethod
    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        """Delete one step, or every step of the task, and return how many were removed."""

    async def close(self) -> None:  # noqa: B027 - optional hook; stores without resources skip it
        """Release backend resources. The default store holds none."""


class InMemoryTaskStepStore(TaskStepStore):
    """Process-local store for tests, with the same rules as the durable stores."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], tuple[str, int]] = {}

    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        key = task_step_ids(task_id, step)
        stored = self._steps.get(key)
        return None if stored is None else task_step_record(*key, *stored)

    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        key = task_step_ids(task_id, step)
        encoded = encode_task_step_result(result)
        stored = self._steps.setdefault(key, (encoded, task_step_saved_at_ms()))
        return task_step_record(*key, *stored)

    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        task_id = task_step_key(task_id, "task_id")
        after = optional_task_step_key(after, "after")
        limit = task_step_list_limit(limit)
        steps = sorted(
            step
            for owner, step in self._steps
            if owner == task_id and (after is None or step > after)
        )
        return [
            task_step_record(task_id, step, *self._steps[(task_id, step)]) for step in steps[:limit]
        ]

    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        task_id = task_step_key(task_id, "task_id")
        if step is not None:
            return 0 if self._steps.pop(task_step_ids(task_id, step), None) is None else 1
        keys = [key for key in self._steps if key[0] == task_id]
        for key in keys:
            del self._steps[key]
        return len(keys)


class TaskSteps:
    """Run and save the steps of one task.

    ``await steps.run("checkout", fn)`` returns the saved result when the step
    already finished, and otherwise calls ``fn`` (sync or async, no
    arguments), saves its result and returns it. Either way the value comes
    from the store, so a first run and a later run get equal values. The result
    must be JSON: dicts with string keys, lists, strings, numbers, booleans and
    ``None``. An exception from ``fn`` saves nothing.
    """

    def __init__(self, store: TaskStepStore, task_id: str) -> None:
        if not isinstance(store, TaskStepStore):
            raise TypeError("store must be a TaskStepStore.")
        self._store = store
        self._task_id = task_step_key(task_id, "task_id")

    @property
    def task_id(self) -> str:
        return self._task_id

    async def run(self, step: str, fn: Callable[[], Awaitable[Any] | Any]) -> Any:
        saved = await self._store.load_step(self._task_id, step)
        if saved is not None:
            return saved.result
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        return (await self._store.save_step(self._task_id, step, value)).result

    async def load(self, step: str) -> TaskStepRecord | None:
        return await self._store.load_step(self._task_id, step)

    async def save(self, step: str, result: Any) -> TaskStepRecord:
        return await self._store.save_step(self._task_id, step, result)

    async def saved(self, *, after: str | None = None, limit: int = 100) -> list[TaskStepRecord]:
        """Return this task's saved steps ordered by step name."""

        return await self._store.list_steps(self._task_id, after=after, limit=limit)

    async def forget(self, step: str) -> bool:
        """Delete one saved step so the next :meth:`run` calls it again."""

        # Validate here: the store reads step=None as "every step of the task".
        step = task_step_key(step, "step")
        return await self._store.delete_steps(self._task_id, step=step) == 1

    async def clear(self) -> int:
        """Delete every saved step of this task, for example after it completes."""

        return await self._store.delete_steps(self._task_id)


def task_steps(store: TaskStepStore, task: Any) -> TaskSteps:
    """Return :class:`TaskSteps` for a :class:`~cayu.tasks.records.Task` or a task id."""

    task_id = task if type(task) is str else getattr(task, "id", None)
    if type(task_id) is not str:
        raise TypeError("task must be a Task or a task id string.")
    return TaskSteps(store, task_id)


__all__ = [
    "TASK_STEP_KEY_MAX_BYTES",
    "TASK_STEP_LIST_MAX",
    "TASK_STEP_RESULT_MAX_BYTES",
    "InMemoryTaskStepStore",
    "TaskStepRecord",
    "TaskStepStore",
    "TaskSteps",
    "encode_task_step_result",
    "optional_task_step_key",
    "task_step_ids",
    "task_step_key",
    "task_steps",
]
