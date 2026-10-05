"""Native external correlation/election records; independent of sessions' JSON."""

SQLITE_EXTERNAL_WAIT_DDL = """
CREATE TABLE IF NOT EXISTS cayu_external_wait_scopes (
    scope TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK (generation BETWEEN 1 AND 9007199254740991),
    limits_json TEXT NOT NULL,
    retired INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0, 1)),
    retirement_json TEXT,
    PRIMARY KEY (scope, generation)
);
CREATE TABLE IF NOT EXISTS cayu_external_waits (
    scope TEXT NOT NULL,
    generation INTEGER NOT NULL,
    correlation_key TEXT NOT NULL,
    source TEXT NOT NULL,
    incarnation TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision BETWEEN 1 AND 9007199254740991),
    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes BETWEEN 2 AND 131072),
    handoff TEXT NOT NULL CHECK (handoff IN ('unbound', 'pending', 'settled', 'excluded')),
    pending_handoff INTEGER NOT NULL CHECK (pending_handoff IN (0, 1)),
    session_id TEXT,
    session_instance_id TEXT,
    record_json TEXT NOT NULL CHECK (
        json_valid(record_json) AND json_type(record_json) = 'object'
        AND length(CAST(record_json AS BLOB)) BETWEEN 1 AND 524288
    ),
    CHECK ((session_id IS NULL) = (session_instance_id IS NULL)),
    PRIMARY KEY (scope, generation, correlation_key),
    FOREIGN KEY (scope, generation) REFERENCES cayu_external_wait_scopes(scope, generation)
);
CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_handoff
    ON cayu_external_waits(scope, generation, handoff, correlation_key);
CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_source
    ON cayu_external_waits(scope, generation, source, correlation_key);
CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_session
    ON cayu_external_waits(session_id, session_instance_id, handoff);
"""

POSTGRES_EXTERNAL_WAIT_DDL = (
    """CREATE TABLE IF NOT EXISTS cayu_external_wait_scopes (
        scope TEXT NOT NULL,
        generation BIGINT NOT NULL CHECK (generation BETWEEN 1 AND 9007199254740991),
        limits_json TEXT NOT NULL,
        retired INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0, 1)),
        retirement_json TEXT,
        PRIMARY KEY (scope, generation)
    )""",
    """CREATE TABLE IF NOT EXISTS cayu_external_waits (
        scope TEXT NOT NULL,
        generation BIGINT NOT NULL,
        correlation_key TEXT NOT NULL,
        source TEXT NOT NULL,
        incarnation TEXT NOT NULL,
        revision BIGINT NOT NULL CHECK (revision BETWEEN 1 AND 9007199254740991),
        reserved_bytes BIGINT NOT NULL CHECK (reserved_bytes BETWEEN 2 AND 131072),
        handoff TEXT NOT NULL CHECK (handoff IN ('unbound', 'pending', 'settled', 'excluded')),
        pending_handoff INTEGER NOT NULL CHECK (pending_handoff IN (0, 1)),
        session_id TEXT,
        session_instance_id TEXT,
        record_json TEXT NOT NULL CHECK (octet_length(record_json) BETWEEN 1 AND 524288),
        CHECK ((session_id IS NULL) = (session_instance_id IS NULL)),
        PRIMARY KEY (scope, generation, correlation_key),
        FOREIGN KEY (scope, generation) REFERENCES cayu_external_wait_scopes(scope, generation)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_handoff "
    "ON cayu_external_waits(scope, generation, handoff, correlation_key)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_source "
    "ON cayu_external_waits(scope, generation, source, correlation_key)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_external_wait_session "
    "ON cayu_external_waits(session_id, session_instance_id, handoff)",
)
