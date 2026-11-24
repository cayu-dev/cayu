"""Schema for durable tool-call preauthorizations."""

from __future__ import annotations

TOOL_CALL_PREAUTHORIZATIONS_TABLE = "cayu_tool_call_preauthorizations"
TOOL_CALL_PREAUTHORIZATION_REVISION = 162

# Indexed columns are projections of ``record_json``, which is the authority.
# ``created_at_ms`` is UTC milliseconds so every backend orders records alike.
SQLITE_TOOL_CALL_PREAUTHORIZATION_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TOOL_CALL_PREAUTHORIZATIONS_TABLE} (
        preauthorization_id TEXT COLLATE BINARY PRIMARY KEY NOT NULL,
        task_id TEXT COLLATE BINARY NOT NULL,
        tool_name TEXT COLLATE BINARY NOT NULL,
        created_at_ms INTEGER NOT NULL,
        record_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS cayu_tool_call_preauthorizations_task
        ON {TOOL_CALL_PREAUTHORIZATIONS_TABLE} (task_id, created_at_ms, preauthorization_id);
"""

POSTGRES_TOOL_CALL_PREAUTHORIZATION_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {TOOL_CALL_PREAUTHORIZATIONS_TABLE} (
        preauthorization_id TEXT COLLATE "C" PRIMARY KEY,
        task_id TEXT COLLATE "C" NOT NULL,
        tool_name TEXT COLLATE "C" NOT NULL,
        created_at_ms BIGINT NOT NULL,
        record_json TEXT NOT NULL
    )""",
    f"""CREATE INDEX IF NOT EXISTS cayu_tool_call_preauthorizations_task
        ON {TOOL_CALL_PREAUTHORIZATIONS_TABLE} (task_id, created_at_ms, preauthorization_id)""",
)


__all__ = [
    "POSTGRES_TOOL_CALL_PREAUTHORIZATION_DDL",
    "SQLITE_TOOL_CALL_PREAUTHORIZATION_DDL",
    "TOOL_CALL_PREAUTHORIZATIONS_TABLE",
    "TOOL_CALL_PREAUTHORIZATION_REVISION",
]
