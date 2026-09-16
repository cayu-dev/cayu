"""Task graph authority is retained independently of session lifetime."""

from cayu.storage._session_closure_sql import POSTGRES_TASK_CLOSURE_GUARD_DDL

SQLITE_TASK_GRAPH_DDL = """
CREATE TABLE IF NOT EXISTS cayu_task_graphs (
    graph_id TEXT PRIMARY KEY, receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json))
);
CREATE TABLE IF NOT EXISTS cayu_task_graph_members (
    task_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
    prerequisites_json TEXT NOT NULL CHECK (json_valid(prerequisites_json)),
    terminal_json TEXT CHECK (terminal_json IS NULL OR json_valid(terminal_json))
);
CREATE INDEX IF NOT EXISTS idx_cayu_task_graph_members_graph ON cayu_task_graph_members(graph_id, task_id);
CREATE TABLE IF NOT EXISTS cayu_task_graph_events (
    graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
    sequence INTEGER NOT NULL CHECK (sequence BETWEEN 1 AND 9007199254740991),
    event_json TEXT NOT NULL CHECK (json_valid(event_json)), PRIMARY KEY (graph_id, sequence)
);
"""

POSTGRES_TASK_GRAPH_DDL = (
    POSTGRES_TASK_CLOSURE_GUARD_DDL,
    "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS graph_id TEXT",
    "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS prerequisite_task_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
    "CREATE TABLE IF NOT EXISTS cayu_task_graphs (graph_id TEXT PRIMARY KEY, receipt_json JSONB NOT NULL)",
    """CREATE TABLE IF NOT EXISTS cayu_task_graph_members (
        task_id TEXT PRIMARY KEY, graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
        prerequisites_json JSONB NOT NULL, terminal_json JSONB
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cayu_task_graph_members_graph ON cayu_task_graph_members(graph_id, task_id)",
    """CREATE TABLE IF NOT EXISTS cayu_task_graph_events (
        graph_id TEXT NOT NULL REFERENCES cayu_task_graphs(graph_id),
        sequence BIGINT NOT NULL CHECK (sequence BETWEEN 1 AND 9007199254740991),
        event_json JSONB NOT NULL, PRIMARY KEY (graph_id, sequence)
    )""",
)
