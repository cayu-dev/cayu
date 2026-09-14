"""Native task scheduling state and task-owned immutable evidence."""

SQLITE_SCHEDULING_DDL = """
CREATE TABLE IF NOT EXISTS cayu_task_schedule_receipts (
    task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE CASCADE,
    operation_id TEXT NOT NULL,
    receipt_json TEXT NOT NULL CHECK (
        json_valid(receipt_json) AND json_type(receipt_json) = 'object'
        AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 32768
    ),
    PRIMARY KEY (task_id, operation_id)
);
CREATE TABLE IF NOT EXISTS cayu_task_schedule_events (
    task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence BETWEEN 1 AND 9007199254740991),
    event_json TEXT NOT NULL CHECK (
        json_valid(event_json) AND json_type(event_json) = 'object'
        AND length(CAST(event_json AS BLOB)) BETWEEN 1 AND 32768
    ),
    PRIMARY KEY (task_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_next_schedule
    ON cayu_tasks(available_at, created_at, id)
    WHERE status = 'pending' AND session_id IS NULL AND available_at IS NOT NULL;
"""

POSTGRES_SCHEDULING_DDL = (
    "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS schedule JSONB "
    "CHECK (schedule IS NULL OR (jsonb_typeof(schedule) = 'object' "
    "AND octet_length(schedule::text) BETWEEN 1 AND 32768))",
    """
    CREATE TABLE IF NOT EXISTS cayu_task_schedule_receipts (
        task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE CASCADE,
        operation_id TEXT NOT NULL,
        receipt_json JSONB NOT NULL CHECK (
            jsonb_typeof(receipt_json) = 'object'
            AND octet_length(receipt_json::text) BETWEEN 1 AND 32768
        ),
        PRIMARY KEY (task_id, operation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_task_schedule_events (
        task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE CASCADE,
        sequence BIGINT NOT NULL CHECK (sequence BETWEEN 1 AND 9007199254740991),
        event_json JSONB NOT NULL CHECK (
            jsonb_typeof(event_json) = 'object'
            AND octet_length(event_json::text) BETWEEN 1 AND 32768
        ),
        PRIMARY KEY (task_id, sequence)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_next_schedule "
    "ON cayu_tasks(available_at, created_at, id) "
    "WHERE status = 'pending' AND session_id IS NULL AND available_at IS NOT NULL",
)
