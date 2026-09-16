"""Task groups share graph ownership and the existing task database."""

SQLITE_TASK_GROUP_DDL = """
CREATE TABLE IF NOT EXISTS cayu_task_groups (
    group_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL UNIQUE REFERENCES cayu_task_graphs(graph_id),
    snapshot_json TEXT NOT NULL CHECK (json_valid(snapshot_json))
);
CREATE TABLE IF NOT EXISTS cayu_task_group_events (
    group_id TEXT NOT NULL REFERENCES cayu_task_groups(group_id),
    sequence INTEGER NOT NULL CHECK (sequence BETWEEN 1 AND 131),
    event_json TEXT NOT NULL CHECK (json_valid(event_json)),
    PRIMARY KEY (group_id, sequence)
);
"""

POSTGRES_TASK_GROUP_DDL = (
    """CREATE TABLE IF NOT EXISTS cayu_task_groups (
        group_id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL UNIQUE REFERENCES cayu_task_graphs(graph_id),
        snapshot_json JSONB NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS cayu_task_group_events (
        group_id TEXT NOT NULL REFERENCES cayu_task_groups(group_id),
        sequence BIGINT NOT NULL CHECK (sequence BETWEEN 1 AND 131),
        event_json JSONB NOT NULL,
        PRIMARY KEY (group_id, sequence)
    )""",
)
