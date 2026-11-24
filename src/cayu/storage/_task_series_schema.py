"""The table where task stores keep recurring task series.

One row per series: its definition, paused state and the occurrence queued last,
as ``TaskSeriesState`` JSON. Occurrences themselves are ordinary task rows.
"""

from __future__ import annotations

#: The storage revision that adds the table. Several revisions are in flight at
#: once; this is the one number to change when renumbering.
TASK_SERIES_REVISION = 161

SQLITE_TASK_SERIES_DDL = """CREATE TABLE IF NOT EXISTS cayu_task_series (
    series_id TEXT COLLATE BINARY PRIMARY KEY,
    state_json TEXT NOT NULL
);"""

POSTGRES_TASK_SERIES_DDL = (
    """CREATE TABLE IF NOT EXISTS cayu_task_series (
    series_id TEXT COLLATE "C" PRIMARY KEY,
    state_json TEXT NOT NULL
)""",
)
