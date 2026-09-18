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

SQLITE_TASK_GROUP_QUIESCENCE_DDL = """
CREATE TABLE IF NOT EXISTS cayu_task_group_retry_lineage (
    task_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
    root_task_id TEXT NOT NULL REFERENCES cayu_task_graph_members(task_id)
);
CREATE INDEX IF NOT EXISTS idx_cayu_group_retry_graph ON cayu_task_group_retry_lineage(graph_id, task_id);
CREATE INDEX IF NOT EXISTS idx_cayu_task_group_barriers ON cayu_task_groups(barrier_status, group_id);
ALTER TABLE cayu_task_group_events RENAME TO cayu_task_group_events_v92;
CREATE TABLE cayu_task_group_events (
    group_id TEXT NOT NULL REFERENCES cayu_task_groups(group_id),
    sequence INTEGER NOT NULL CHECK (sequence BETWEEN 1 AND 9007199254740991),
    event_json TEXT NOT NULL CHECK (json_valid(event_json)),
    PRIMARY KEY (group_id, sequence)
);
INSERT INTO cayu_task_group_events SELECT * FROM cayu_task_group_events_v92;
DROP TABLE cayu_task_group_events_v92;
CREATE TABLE IF NOT EXISTS cayu_task_group_resolutions (
    group_id TEXT NOT NULL REFERENCES cayu_task_groups(group_id),
    idempotency_key TEXT NOT NULL, request_sha256 TEXT NOT NULL,
    snapshot_json TEXT NOT NULL CHECK (json_valid(snapshot_json)),
    PRIMARY KEY (group_id, idempotency_key)
);
"""

POSTGRES_TASK_GROUP_QUIESCENCE_DDL = (
    """CREATE TABLE IF NOT EXISTS cayu_task_group_retry_lineage (
        task_id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
        root_task_id TEXT NOT NULL REFERENCES cayu_task_graph_members(task_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cayu_group_retry_graph ON cayu_task_group_retry_lineage(graph_id, task_id)",
    "ALTER TABLE cayu_task_groups ADD COLUMN IF NOT EXISTS barrier_status TEXT NOT NULL DEFAULT 'not_requested'",
    "ALTER TABLE cayu_task_groups ADD COLUMN IF NOT EXISTS barrier_deadline TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS idx_cayu_task_group_barriers ON cayu_task_groups(barrier_status, group_id)",
    "ALTER TABLE cayu_task_group_events DROP CONSTRAINT IF EXISTS cayu_task_group_events_sequence_check",
    "ALTER TABLE cayu_task_group_events ADD CHECK (sequence BETWEEN 1 AND 9007199254740991)",
    """CREATE TABLE IF NOT EXISTS cayu_task_group_resolutions (
        group_id TEXT NOT NULL REFERENCES cayu_task_groups(group_id),
        idempotency_key TEXT NOT NULL, request_sha256 TEXT NOT NULL,
        snapshot_json JSONB NOT NULL,
        PRIMARY KEY (group_id, idempotency_key)
    )""",
)
