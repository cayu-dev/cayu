"""Storage-retention audit records join the additive schema history.

One run row records the policy and, once the run finishes, its summary. One
entry row per changed item is written in the same transaction as the change,
so an interrupted run still leaves an exact record of what it removed. Neither
table references a session: the audit outlives the records it describes.
"""

RETENTION_AUDIT_REVISION = 118

SQLITE_RETENTION_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS cayu_storage_retention_runs (
    audit_id TEXT COLLATE BINARY PRIMARY KEY,
    store_kind TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('compact', 'delete')),
    state TEXT NOT NULL CHECK (state IN ('started', 'completed')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    policy_json TEXT NOT NULL CHECK (json_valid(policy_json)),
    summary_json TEXT CHECK (summary_json IS NULL OR json_valid(summary_json)),
    CHECK ((state = 'completed') = (completed_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_cayu_storage_retention_runs_started
    ON cayu_storage_retention_runs(store_kind, started_at);
CREATE TABLE IF NOT EXISTS cayu_storage_retention_entries (
    audit_id TEXT COLLATE BINARY NOT NULL
        REFERENCES cayu_storage_retention_runs(audit_id) ON DELETE CASCADE,
    item_id TEXT COLLATE BINARY NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('compact', 'delete')),
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    counts_json TEXT NOT NULL CHECK (json_valid(counts_json)),
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (audit_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_cayu_storage_retention_entries_item
    ON cayu_storage_retention_entries(item_id);
"""

POSTGRES_RETENTION_AUDIT_DDL = (
    """
    CREATE TABLE IF NOT EXISTS cayu_storage_retention_runs (
        audit_id TEXT PRIMARY KEY,
        store_kind TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('compact', 'delete')),
        state TEXT NOT NULL CHECK (state IN ('started', 'completed')),
        started_at TIMESTAMPTZ NOT NULL,
        completed_at TIMESTAMPTZ,
        policy_json JSONB NOT NULL,
        summary_json JSONB,
        CHECK ((state = 'completed') = (completed_at IS NOT NULL))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_storage_retention_runs_started "
    "ON cayu_storage_retention_runs(store_kind, started_at)",
    """
    CREATE TABLE IF NOT EXISTS cayu_storage_retention_entries (
        audit_id TEXT NOT NULL
            REFERENCES cayu_storage_retention_runs(audit_id) ON DELETE CASCADE,
        item_id TEXT NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('compact', 'delete')),
        bytes BIGINT NOT NULL CHECK (bytes >= 0),
        counts_json JSONB NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (audit_id, item_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_storage_retention_entries_item "
    "ON cayu_storage_retention_entries(item_id)",
)
