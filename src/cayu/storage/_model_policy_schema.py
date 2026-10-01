"""Application-policy state joins the existing additive schema history."""

SQLITE_MODEL_POLICY_DDL = """
CREATE TABLE IF NOT EXISTS cayu_model_policy_state (
    binding_id TEXT PRIMARY KEY,
    binding BLOB NOT NULL CHECK (length(binding) BETWEEN 1 AND 65536),
    owner TEXT,
    expires REAL NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    state BLOB NOT NULL CHECK (length(state) BETWEEN 1 AND 16777216)
);
"""

POSTGRES_MODEL_POLICY_DDL = (
    """
CREATE TABLE IF NOT EXISTS cayu_model_policy_state (
    binding_id TEXT PRIMARY KEY,
    binding BYTEA NOT NULL CHECK (octet_length(binding) BETWEEN 1 AND 65536),
    owner TEXT,
    expires DOUBLE PRECISION NOT NULL,
    revision BIGINT NOT NULL CHECK (revision >= 0),
    state BYTEA NOT NULL CHECK (octet_length(state) BETWEEN 1 AND 16777216)
)
""",
)
