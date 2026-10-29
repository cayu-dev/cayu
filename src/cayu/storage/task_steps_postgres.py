"""PostgreSQL store for saved task step results."""

from __future__ import annotations

from typing import Any

from psycopg.rows import tuple_row

from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage._task_step_schema import TASK_STEPS_REVISION, TASK_STEPS_TABLE
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

_SELECT = f"SELECT task_id, step, result_json, saved_at_ms FROM {TASK_STEPS_TABLE}"


class PostgresTaskStepStore(_PostgresStoreBase, TaskStepStore):
    """Saved task step results shared by worker processes on PostgreSQL.

    Use the task store's DSN to keep a task's steps beside the task. The table
    joins Cayu's shared revision history; the default ``schema_mode`` is
    ``VALIDATE`` and ``cayu storage migrate`` creates or upgrades it.
    """

    _min_required_revision = TASK_STEPS_REVISION

    async def load_step(self, task_id: str, step: str) -> TaskStepRecord | None:
        key = task_step_ids(task_id, step)
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(f"{_SELECT} WHERE task_id = %s AND step = %s", key)
            row = await cur.fetchone()
            await conn.commit()
        return None if row is None else task_step_record(*row)

    async def save_step(self, task_id: str, step: str, result: Any) -> TaskStepRecord:
        key = task_step_ids(task_id, step)
        encoded = encode_task_step_result(result)
        # The no-op update locks and returns the existing row, so a concurrent
        # delete can't leave the save with nothing to read back.
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(
                        f"INSERT INTO {TASK_STEPS_TABLE} AS existing "
                        "(task_id, step, result_json, saved_at_ms) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (task_id, step) DO UPDATE "
                        "SET result_json = existing.result_json "
                        "RETURNING task_id, step, result_json, saved_at_ms",
                        (*key, encoded, task_step_saved_at_ms()),
                    )
                    row = await cur.fetchone()
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        if row is None:  # pragma: no cover - INSERT ... RETURNING always returns the row
            raise RuntimeError("Saving a task step returned no row.")
        return task_step_record(*row)

    async def list_steps(
        self, task_id: str, *, after: str | None = None, limit: int = 100
    ) -> list[TaskStepRecord]:
        task_id = task_step_key(task_id, "task_id")
        after = optional_task_step_key(after, "after")
        limit = task_step_list_limit(limit)
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(
                f"{_SELECT} WHERE task_id = %s AND (%s::text IS NULL OR step > %s::text) "
                "ORDER BY step LIMIT %s",
                (task_id, after, after, limit),
            )
            rows = await cur.fetchall()
            await conn.commit()
        return [task_step_record(*row) for row in rows]

    async def delete_steps(self, task_id: str, *, step: str | None = None) -> int:
        task_id = task_step_key(task_id, "task_id")
        step = optional_task_step_key(step, "step")
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(
                        f"DELETE FROM {TASK_STEPS_TABLE} "
                        "WHERE task_id = %s AND (%s::text IS NULL OR step = %s::text)",
                        (task_id, step, step),
                    )
                    deleted = cur.rowcount
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return deleted


__all__ = ["PostgresTaskStepStore"]
