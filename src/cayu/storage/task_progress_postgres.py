"""PostgreSQL store for task progress counters."""

from __future__ import annotations

import json

from psycopg.rows import tuple_row

from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage._task_progress_schema import TASK_PROGRESS_REVISION, TASK_PROGRESS_TABLE
from cayu.tasks.progress import (
    TaskProgress,
    TaskProgressStore,
    encode_task_progress_counts,
    merged_task_progress,
    task_progress_key,
    task_progress_record,
    task_progress_updated_at_ms,
)

_SELECT = f"SELECT task_id, done, total, counts_json, updated_at_ms FROM {TASK_PROGRESS_TABLE}"


class PostgresTaskProgressStore(_PostgresStoreBase, TaskProgressStore):
    """Task progress shared by worker processes on PostgreSQL.

    Use the task store's DSN to keep a task's progress beside the task. The table
    joins Cayu's shared revision history; the default ``schema_mode`` is ``VALIDATE``
    and ``cayu storage migrate`` creates or upgrades it.
    """

    _min_required_revision = TASK_PROGRESS_REVISION

    async def load_progress(self, task_id: str) -> TaskProgress | None:
        task_id = task_progress_key(task_id, "task_id")
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(f"{_SELECT} WHERE task_id = %s", (task_id,))
            row = await cur.fetchone()
            await conn.commit()
        return None if row is None else task_progress_record(*row)

    async def _write_progress(self, task_id: str, **change) -> TaskProgress:
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    # Lock the row, or the key of one not written yet, for this write.
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"{TASK_PROGRESS_TABLE}:{task_id}",),
                    )
                    await cur.execute(f"{_SELECT} WHERE task_id = %s", (task_id,))
                    row = await cur.fetchone()
                    current = None if row is None else (row[1], row[2], json.loads(row[3]))
                    done, total, counts = merged_task_progress(current, **change)
                    stored = (
                        task_id,
                        done,
                        total,
                        encode_task_progress_counts(counts),
                        task_progress_updated_at_ms(),
                    )
                    await cur.execute(
                        f"INSERT INTO {TASK_PROGRESS_TABLE} "
                        "(task_id, done, total, counts_json, updated_at_ms) "
                        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (task_id) DO UPDATE SET "
                        "done = EXCLUDED.done, total = EXCLUDED.total, "
                        "counts_json = EXCLUDED.counts_json, "
                        "updated_at_ms = EXCLUDED.updated_at_ms",
                        stored,
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return task_progress_record(*stored)

    async def delete_progress(self, task_id: str) -> bool:
        task_id = task_progress_key(task_id, "task_id")
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor() as cur:
                    # The lock writes take, so a delete can't interleave with an addition.
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"{TASK_PROGRESS_TABLE}:{task_id}",),
                    )
                    await cur.execute(
                        f"DELETE FROM {TASK_PROGRESS_TABLE} WHERE task_id = %s", (task_id,)
                    )
                    deleted = cur.rowcount
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return deleted > 0


__all__ = ["PostgresTaskProgressStore"]
