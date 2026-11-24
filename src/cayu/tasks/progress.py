"""Progress counters for a task that works through many units.

A handler reviewing 40 cases reports how far it got: ``await store.add_progress(task.id,
done=1, counts={"approved": 1})`` after each case, with ``set_progress(task.id,
total=40)`` once it knows the total. Anyone can read ``load_progress(task.id)``:
"12 of 40 done: 10 approved, 2 failed", and when it last changed.

The parts stand alone. :class:`TaskProgressStore` is the storage contract,
:class:`InMemoryTaskProgressStore` is a process-local store for tests, and
``SQLiteTaskProgressStore`` and ``PostgresTaskProgressStore`` in ``cayu.storage`` keep
progress in its own table in the task store's database. Nothing in the task worker
requires a progress store, and the store doesn't check that the task exists or who
runs it; a task's progress is deleted explicitly.

``add_progress`` adds to the stored counts in one transaction, so several handlers
working through one task's units can each report their own. ``set_progress`` replaces
the fields it is given. Counts are whole numbers; there are at most
:data:`TASK_PROGRESS_MAX_COUNTS` outcome names.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank

#: The most outcome names one task's ``counts`` holds.
TASK_PROGRESS_MAX_COUNTS = 32
#: Longest task id or outcome name, in UTF-8 bytes.
TASK_PROGRESS_KEY_MAX_BYTES = 512


class TaskProgress(BaseModel):
    """How far one task got: ``done`` of ``total`` units, by outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    done: int = Field(ge=0)
    #: ``None`` until the handler says how many units there are.
    total: int | None = Field(default=None, ge=0)
    counts: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime


def task_progress_key(value: object, field_name: str) -> str:
    """Validate a task id or outcome name the way every store does."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > TASK_PROGRESS_KEY_MAX_BYTES:
        raise ValueError(f"{field_name} must not exceed {TASK_PROGRESS_KEY_MAX_BYTES} UTF-8 bytes.")
    return value


def _count(value: object, field_name: str, *, signed: bool = False) -> int:
    low = -MAX_DURABLE_JSON_INTEGER if signed else 0
    if type(value) is not int or not low <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must be a whole number from {low} to the durable limit.")
    return value


def task_progress_counts(value: object, *, signed: bool = False) -> dict[str, int]:
    """Validate an outcome-to-count mapping."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("counts must map outcome names to whole numbers.")
    if len(value) > TASK_PROGRESS_MAX_COUNTS:
        raise ValueError(f"counts must name at most {TASK_PROGRESS_MAX_COUNTS} outcomes.")
    return {
        task_progress_key(name, "counts key"): _count(count, f"counts[{name!r}]", signed=signed)
        for name, count in value.items()
    }


def merged_task_progress(
    current: tuple[int, int | None, dict[str, int]] | None,
    *,
    set_done: int | None = None,
    set_total: int | None = None,
    set_counts: Mapping[str, int] | None = None,
    add_done: int = 0,
    add_counts: Mapping[str, int] | None = None,
) -> tuple[int, int | None, dict[str, int]]:
    """The stored ``(done, total, counts)`` after one write; every store applies this."""

    done, total, counts = current if current is not None else (0, None, {})
    counts = dict(counts)
    if set_done is not None:
        done = set_done
    if set_total is not None:
        total = set_total
    if set_counts is not None:
        counts = dict(set_counts)
    done = _count(done + add_done, "done")
    for name, delta in (add_counts or {}).items():
        counts[name] = _count(counts.get(name, 0) + delta, f"counts[{name!r}]")
    if len(counts) > TASK_PROGRESS_MAX_COUNTS:
        raise ValueError(f"counts must name at most {TASK_PROGRESS_MAX_COUNTS} outcomes.")
    return done, total, counts


def encode_task_progress_counts(counts: Mapping[str, int]) -> str:
    return json.dumps(dict(sorted(counts.items())), separators=(",", ":"), ensure_ascii=False)


def task_progress_updated_at_ms() -> int:
    """Store-side update time in UTC milliseconds, the precision every backend keeps."""

    return int(datetime.now(UTC).timestamp() * 1000)


def task_progress_record(
    task_id: str, done: int, total: int | None, counts_json: str, updated_at_ms: int
) -> TaskProgress:
    """Rebuild a record from its stored columns."""

    return TaskProgress(
        task_id=task_id,
        done=done,
        total=total,
        counts=json.loads(counts_json),
        updated_at=datetime.fromtimestamp(updated_at_ms / 1000, tz=UTC),
    )


class TaskProgressStore(ABC):
    """Storage contract for task progress counters."""

    @abstractmethod
    async def load_progress(self, task_id: str) -> TaskProgress | None:
        """The task's progress, or ``None`` when nothing was reported."""

    @abstractmethod
    async def _write_progress(
        self,
        task_id: str,
        *,
        set_done: int | None,
        set_total: int | None,
        set_counts: dict[str, int] | None,
        add_done: int,
        add_counts: dict[str, int],
    ) -> TaskProgress:
        """Apply ``merged_task_progress`` to the stored row in one transaction."""

    @abstractmethod
    async def delete_progress(self, task_id: str) -> bool:
        """Delete the task's progress; ``False`` when there was none."""

    async def set_progress(
        self,
        task_id: str,
        *,
        done: int | None = None,
        total: int | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> TaskProgress:
        """Replace the fields given (``counts`` as a whole) and return the progress."""

        return await self._write_progress(
            task_progress_key(task_id, "task_id"),
            set_done=None if done is None else _count(done, "done"),
            set_total=None if total is None else _count(total, "total"),
            set_counts=None if counts is None else task_progress_counts(counts),
            add_done=0,
            add_counts={},
        )

    async def add_progress(
        self, task_id: str, *, done: int = 0, counts: Mapping[str, int] | None = None
    ) -> TaskProgress:
        """Add ``done`` and each of ``counts`` to the stored values, atomically."""

        return await self._write_progress(
            task_progress_key(task_id, "task_id"),
            set_done=None,
            set_total=None,
            set_counts=None,
            add_done=_count(done, "done", signed=True),
            add_counts=task_progress_counts(counts, signed=True),
        )

    async def close(self) -> None:  # noqa: B027 - optional hook; stores without resources skip it
        """Release backend resources. The default store holds none."""


class InMemoryTaskProgressStore(TaskProgressStore):
    """Process-local store for tests, with the same rules as the durable stores."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[int, int | None, str, int]] = {}

    async def load_progress(self, task_id: str) -> TaskProgress | None:
        task_id = task_progress_key(task_id, "task_id")
        row = self._rows.get(task_id)
        return None if row is None else task_progress_record(task_id, *row)

    async def _write_progress(self, task_id: str, **change) -> TaskProgress:
        row = self._rows.get(task_id)
        current = None if row is None else (row[0], row[1], json.loads(row[2]))
        done, total, counts = merged_task_progress(current, **change)
        self._rows[task_id] = (
            done,
            total,
            encode_task_progress_counts(counts),
            task_progress_updated_at_ms(),
        )
        return task_progress_record(task_id, *self._rows[task_id])

    async def delete_progress(self, task_id: str) -> bool:
        return self._rows.pop(task_progress_key(task_id, "task_id"), None) is not None


__all__ = [
    "TASK_PROGRESS_MAX_COUNTS",
    "InMemoryTaskProgressStore",
    "TaskProgress",
    "TaskProgressStore",
]
