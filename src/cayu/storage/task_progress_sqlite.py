"""SQLite store for task progress counters."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage._task_progress_schema import TASK_PROGRESS_REVISION, TASK_PROGRESS_TABLE
from cayu.storage.targets import require_sqlite_store_allowed
from cayu.tasks.progress import (
    TaskProgress,
    TaskProgressStore,
    encode_task_progress_counts,
    merged_task_progress,
    task_progress_key,
    task_progress_record,
    task_progress_updated_at_ms,
)

_T = TypeVar("_T")

_SELECT = f"SELECT task_id, done, total, counts_json, updated_at_ms FROM {TASK_PROGRESS_TABLE}"


class SQLiteTaskProgressStore(TaskProgressStore):
    """Task progress on SQLite, sharing Cayu's schema revision history.

    Point it at the task store's database file to keep a task's progress beside the
    task. Like the other SQLite stores, ``CREATE`` initializes an empty database and
    otherwise validates; an older database requires ``cayu storage migrate``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        require_sqlite_store_allowed("SQLiteTaskProgressStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteTaskProgressStore path must be a nonblank path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._lock = TimedStoreLock()
        self._closed = False
        self._connection = sqlite_connection.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection, schema_mode, app_min_supported=TASK_PROGRESS_REVISION
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
            raise RuntimeError("SQLiteTaskProgressStore is closed.")
        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    async def load_progress(self, task_id: str) -> TaskProgress | None:
        task_id = task_progress_key(task_id, "task_id")

        def load(connection: sqlite3.Connection) -> TaskProgress | None:
            with sqlite_connection._transaction(connection, begin_immediate=False):
                row = connection.execute(f"{_SELECT} WHERE task_id = ?", (task_id,)).fetchone()
            return None if row is None else task_progress_record(*row)

        return await self._run(load)

    async def _write_progress(self, task_id: str, **change) -> TaskProgress:
        def write(connection: sqlite3.Connection) -> TaskProgress:
            with sqlite_connection._transaction(connection):
                row = connection.execute(f"{_SELECT} WHERE task_id = ?", (task_id,)).fetchone()
                current = None if row is None else (row[1], row[2], json.loads(row[3]))
                done, total, counts = merged_task_progress(current, **change)
                stored = (
                    task_id,
                    done,
                    total,
                    encode_task_progress_counts(counts),
                    task_progress_updated_at_ms(),
                )
                connection.execute(
                    f"INSERT INTO {TASK_PROGRESS_TABLE} "
                    "(task_id, done, total, counts_json, updated_at_ms) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (task_id) DO UPDATE SET done = excluded.done, "
                    "total = excluded.total, counts_json = excluded.counts_json, "
                    "updated_at_ms = excluded.updated_at_ms",
                    stored,
                )
            return task_progress_record(*stored)

        return await self._run(write)

    async def delete_progress(self, task_id: str) -> bool:
        task_id = task_progress_key(task_id, "task_id")

        def delete(connection: sqlite3.Connection) -> bool:
            with sqlite_connection._transaction(connection):
                cursor = connection.execute(
                    f"DELETE FROM {TASK_PROGRESS_TABLE} WHERE task_id = ?", (task_id,)
                )
            return cursor.rowcount > 0

        return await self._run(delete)


__all__ = ["SQLiteTaskProgressStore"]
