"""SQLite store for saved task step results."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage._task_step_schema import TASK_STEPS_REVISION, TASK_STEPS_TABLE
from cayu.storage.targets import require_sqlite_store_allowed
from cayu.tasks.steps import (
    TaskStepRecord,
    TaskStepStore,
    encode_task_step_result,
    optional_task_step_key,
    task_step_ids,
    task_step_key,
    task_step_list_limit,
    task_step_record,
    task_step_saved_at_ms,
)

_T = TypeVar("_T")

_SELECT = f"SELECT task_id, step, result_json, saved_at_ms FROM {TASK_STEPS_TABLE}"


class SQLiteTaskStepStore(TaskStepStore):
    """Saved task step results on SQLite, sharing Cayu's schema revision history.

    Point it at the task store's database file to keep a task's steps beside
    the task. Like the other SQLite stores, ``CREATE`` initializes an empty
    database and otherwise validates; an older database requires
    ``cayu storage migrate``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        require_sqlite_store_allowed("SQLiteTaskStepStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteTaskStepStore path must be a nonblank path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._lock = TimedStoreLock()
        self._closed = False
        self._connection = sqlite_connection.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection, schema_mode, app_min_supported=TASK_STEPS_REVISION
            )
        except BaseException:
            self._connection.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, lambda connection: connection.close()
        )

    async def _run(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        if self._closed:
            raise RuntimeError("SQLiteTaskStepStore is closed.")
        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        key = task_step_ids(task_id, step)

        def load(connection: sqlite3.Connection) -> TaskStepRecord | None:
            with sqlite_connection._transaction(connection, begin_immediate=False):
                row = connection.execute(
                    f"{_SELECT} WHERE task_id = ? AND step = ?", key
                ).fetchone()
            return None if row is None else task_step_record(*row)

        return await self._run(load)

    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        key = task_step_ids(task_id, step)
        encoded = encode_task_step_result(result)

        def save(connection: sqlite3.Connection) -> TaskStepRecord:
            with sqlite_connection._transaction(connection):
                connection.execute(
                    f"INSERT INTO {TASK_STEPS_TABLE} (task_id, step, result_json, saved_at_ms) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (task_id, step) DO NOTHING",
                    (*key, encoded, task_step_saved_at_ms()),
                )
                row = connection.execute(
                    f"{_SELECT} WHERE task_id = ? AND step = ?", key
                ).fetchone()
            return task_step_record(*row)

        return await self._run(save)

    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        task_id = task_step_key(task_id, "task_id")
        after = optional_task_step_key(after, "after")
        limit = task_step_list_limit(limit)

        def list_rows(connection: sqlite3.Connection) -> list[TaskStepRecord]:
            with sqlite_connection._transaction(connection, begin_immediate=False):
                rows = connection.execute(
                    f"{_SELECT} WHERE task_id = ? AND (? IS NULL OR step > ?) "
                    "ORDER BY step LIMIT ?",
                    (task_id, after, after, limit),
                ).fetchall()
            return [task_step_record(*row) for row in rows]

        return await self._run(list_rows)

    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        task_id = task_step_key(task_id, "task_id")
        step = optional_task_step_key(step, "step")

        def delete(connection: sqlite3.Connection) -> int:
            with sqlite_connection._transaction(connection):
                cursor = connection.execute(
                    f"DELETE FROM {TASK_STEPS_TABLE} WHERE task_id = ? AND (? IS NULL OR step = ?)",
                    (task_id, step, step),
                )
            return cursor.rowcount

        return await self._run(delete)


__all__ = ["SQLiteTaskStepStore"]
