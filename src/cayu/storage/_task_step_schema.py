"""Revision-136 schema for saved task step results."""

from __future__ import annotations

TASK_STEPS_TABLE = "cayu_task_steps"
TASK_STEPS_REVISION = 136

# One row per (task, step). ``result_json`` is the canonical compact JSON text,
# kept as text on both backends so a result reads back byte for byte. The row
# has no foreign key to the task: step stores are optional and may live beside
# any task store, and a task's steps are deleted explicitly.
SQLITE_TASK_STEP_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TASK_STEPS_TABLE} (
        task_id TEXT COLLATE BINARY NOT NULL,
        step TEXT COLLATE BINARY NOT NULL,
        result_json TEXT NOT NULL,
        saved_at_ms INTEGER NOT NULL,
        PRIMARY KEY (task_id, step)
    );
"""

POSTGRES_TASK_STEP_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {TASK_STEPS_TABLE} (
        task_id TEXT COLLATE "C" NOT NULL,
        step TEXT COLLATE "C" NOT NULL,
        result_json TEXT NOT NULL,
        saved_at_ms BIGINT NOT NULL,
        PRIMARY KEY (task_id, step)
    )""",
)


__all__ = [
    "POSTGRES_TASK_STEP_DDL",
    "SQLITE_TASK_STEP_DDL",
    "TASK_STEPS_REVISION",
    "TASK_STEPS_TABLE",
]
