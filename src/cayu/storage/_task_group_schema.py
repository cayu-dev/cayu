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


# Revision 146: a retained group execution obligation records the claim
# generation that started it. The migration runs with every worker stopped, so no
# holder will acknowledge its execution again with its own generation; the backfill
# copies each member's current generation, which yields the same planner and
# owner-lost decisions as the exact one. The row is the execution's claim; or
# reclaim's fence of it at generation N+1, which an owner-lost settlement of the
# current row still matches (a drain by the stopped claim N no longer would); or a
# later claim at N+2 or more, which already had to settle the retained execution.
# A deleted member's execution is settled, because deleting a group member
# requires a quiescent group whose executions are all settled. It records the
# initial generation, which no claim can present again.
_DELETED_MEMBER_GENERATION = 0


def _sqlite_execution_generation_backfill(table: str) -> str:
    return f"""
UPDATE {table} SET snapshot_json = json_set(
    snapshot_json,
    '$.quiescence.executions',
    json((
        SELECT json_group_array(json(item)) FROM (
            SELECT json_set(
                execution.value,
                '$.claim_generation',
                COALESCE(task.claim_generation, {_DELETED_MEMBER_GENERATION})
            ) AS item
            FROM json_each({table}.snapshot_json, '$.quiescence.executions') AS execution
            LEFT JOIN cayu_tasks AS task
                ON task.id = json_extract(execution.value, '$.task_id')
            ORDER BY execution.key
        )
    ))
)
WHERE json_array_length(snapshot_json, '$.quiescence.executions') > 0;
"""


def _postgres_execution_generation_backfill(table: str) -> str:
    return f"""UPDATE {table} AS retained SET snapshot_json = jsonb_set(
        retained.snapshot_json,
        '{{quiescence,executions}}',
        (
            SELECT jsonb_agg(
                execution.value
                    || jsonb_build_object(
                        'claim_generation',
                        COALESCE(task.claim_generation, {_DELETED_MEMBER_GENERATION})
                    )
                ORDER BY execution.position
            )
            FROM jsonb_array_elements(retained.snapshot_json -> 'quiescence' -> 'executions')
                WITH ORDINALITY AS execution(value, position)
            LEFT JOIN cayu_tasks AS task ON task.id = execution.value ->> 'task_id'
        )
    )
    WHERE jsonb_array_length(retained.snapshot_json -> 'quiescence' -> 'executions') > 0"""


_GENERATION_BACKFILL_TABLES = ("cayu_task_groups", "cayu_task_group_resolutions")
SQLITE_GROUP_EXECUTION_GENERATION_BACKFILL = "".join(
    _sqlite_execution_generation_backfill(table) for table in _GENERATION_BACKFILL_TABLES
)
POSTGRES_GROUP_EXECUTION_GENERATION_BACKFILL = tuple(
    _postgres_execution_generation_backfill(table) for table in _GENERATION_BACKFILL_TABLES
)
