"""Independent completion-evaluation runs join the additive schema history.

Rows reference their proposal with ``ON DELETE CASCADE`` so task closure can
remove a proposal and its evaluation receipts in any dependency order.
"""

SQLITE_COMPLETION_EVALUATION_DDL = """
CREATE TABLE IF NOT EXISTS cayu_completion_evaluation_runs (
    effect_id TEXT NOT NULL PRIMARY KEY,
    proposal_id TEXT NOT NULL
        REFERENCES cayu_completion_proposals(proposal_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
    run_ordinal INTEGER NOT NULL CHECK (run_ordinal BETWEEN 1 AND 16),
    request_sha256 TEXT NOT NULL CHECK (
        length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    settled INTEGER NOT NULL CHECK (settled IN (0, 1)),
    record_json TEXT NOT NULL CHECK (json_valid(record_json)),
    UNIQUE (proposal_id, run_ordinal)
);
CREATE INDEX IF NOT EXISTS idx_cayu_completion_evaluation_runs_task
    ON cayu_completion_evaluation_runs(task_id);
"""

POSTGRES_COMPLETION_EVALUATION_DDL = (
    """
    CREATE TABLE IF NOT EXISTS cayu_completion_evaluation_runs (
        effect_id TEXT PRIMARY KEY,
        proposal_id TEXT NOT NULL
            REFERENCES cayu_completion_proposals(proposal_id) ON DELETE CASCADE,
        task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
        run_ordinal INTEGER NOT NULL CHECK (run_ordinal BETWEEN 1 AND 16),
        request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
        settled BOOLEAN NOT NULL,
        record_json JSONB NOT NULL,
        UNIQUE (proposal_id, run_ordinal)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_completion_evaluation_runs_task "
    "ON cayu_completion_evaluation_runs(task_id)",
)
