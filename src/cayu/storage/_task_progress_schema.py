"""Revision-160 schema for task progress counters."""

from __future__ import annotations

TASK_PROGRESS_TABLE = "cayu_task_progress"
TASK_PROGRESS_REVISION = 160

# One row per task. ``counts_json`` is a compact JSON object of outcome counts. The
# row has no foreign key to the task: progress stores are optional and may live
# beside any task store, and a task's progress is deleted explicitly.
SQLITE_TASK_PROGRESS_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TASK_PROGRESS_TABLE} (
        task_id TEXT COLLATE BINARY PRIMARY KEY,
        done INTEGER NOT NULL,
        total INTEGER,
        counts_json TEXT NOT NULL,
        updated_at_ms INTEGER NOT NULL
    );
"""

POSTGRES_TASK_PROGRESS_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {TASK_PROGRESS_TABLE} (
        task_id TEXT COLLATE "C" PRIMARY KEY,
        done BIGINT NOT NULL,
        total BIGINT,
        counts_json TEXT NOT NULL,
        updated_at_ms BIGINT NOT NULL
    )""",
)


__all__ = [
    "POSTGRES_TASK_PROGRESS_DDL",
    "SQLITE_TASK_PROGRESS_DDL",
    "TASK_PROGRESS_REVISION",
    "TASK_PROGRESS_TABLE",
]
