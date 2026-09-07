"""Indexes and deletion invalidation for authoritative incremental accounting."""

SQLITE_ACCOUNTING_DDL = """
CREATE INDEX IF NOT EXISTS idx_cayu_events_cost_attempt
ON cayu_events(session_id COLLATE BINARY,
    substr(json_extract(payload_json, '$.model_attempt_id'), 1, 128), event_type DESC, sequence)
WHERE event_type IN ('model.completed', 'model.hosted_tool_call');
CREATE INDEX IF NOT EXISTS idx_cayu_events_cost_sequence ON cayu_events(sequence)
WHERE event_type IN ('model.completed', 'model.hosted_tool_call');
CREATE TABLE IF NOT EXISTS cayu_accounting_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    generation INTEGER NOT NULL CHECK (generation BETWEEN 0 AND 9007199254740991)
);
INSERT OR IGNORE INTO cayu_accounting_state(singleton, generation) VALUES (1, 0);
CREATE TRIGGER IF NOT EXISTS cayu_accounting_delete_generation
AFTER DELETE ON cayu_events
WHEN OLD.event_type IN ('model.completed', 'model.hosted_tool_call', 'tool.call.started')
BEGIN
    UPDATE cayu_accounting_state SET generation = generation + 1 WHERE singleton = 1;
END;
"""

POSTGRES_ACCOUNTING_DDL: tuple[str, ...] = (
    """CREATE INDEX IF NOT EXISTS idx_cayu_events_cost_attempt
    ON cayu_events(session_id COLLATE "C",
        (left(event -> 'payload' ->> 'model_attempt_id', 128)) COLLATE "C", event_type DESC, sequence)
    WHERE event_type IN ('model.completed', 'model.hosted_tool_call')""",
    """CREATE INDEX IF NOT EXISTS idx_cayu_events_cost_sequence ON cayu_events(sequence)
    WHERE event_type IN ('model.completed', 'model.hosted_tool_call')""",
    """CREATE TABLE IF NOT EXISTS cayu_accounting_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        generation BIGINT NOT NULL CHECK (generation BETWEEN 0 AND 9007199254740991)
    )""",
    "INSERT INTO cayu_accounting_state(singleton, generation) VALUES (1, 0) ON CONFLICT DO NOTHING",
    """CREATE OR REPLACE FUNCTION cayu_advance_accounting_generation()
    RETURNS TRIGGER LANGUAGE plpgsql AS $cayu_accounting$
    BEGIN
        IF EXISTS (SELECT 1 FROM cayu_removed_accounting_events
                   WHERE event_type IN ('model.completed', 'model.hosted_tool_call', 'tool.call.started')) THEN
            UPDATE cayu_accounting_state SET generation = generation + 1 WHERE singleton = 1;
        END IF;
        RETURN NULL;
    END;
    $cayu_accounting$""",
    "DROP TRIGGER IF EXISTS cayu_accounting_delete_generation ON cayu_events",
    """CREATE TRIGGER cayu_accounting_delete_generation AFTER DELETE ON cayu_events
    REFERENCING OLD TABLE AS cayu_removed_accounting_events
    FOR EACH STATEMENT EXECUTE FUNCTION cayu_advance_accounting_generation()""",
)
