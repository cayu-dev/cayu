"""Revision-119 schema for durable tool authorization grants."""

from __future__ import annotations

TOOL_AUTHORIZATION_GRANTS_TABLE = "cayu_tool_authorization_grants"
TOOL_AUTHORIZATION_GRANT_USES_TABLE = "cayu_tool_authorization_grant_uses"
TOOL_AUTHORIZATION_GRANT_REVISION = 119

# Indexed columns are projections of ``record_json``, which is the authority.
# Timestamps are UTC milliseconds so every backend orders and round-trips
# records identically.
SQLITE_TOOL_AUTHORIZATION_GRANT_DDL = f"""
    CREATE TABLE IF NOT EXISTS {TOOL_AUTHORIZATION_GRANTS_TABLE} (
        grant_id TEXT PRIMARY KEY NOT NULL,
        policy_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'workspace')),
        scope_id TEXT NOT NULL,
        origin_session_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('staged', 'active', 'abandoned', 'revoked')),
        created_at_ms INTEGER NOT NULL,
        record_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS cayu_tool_authorization_grants_scope
        ON {TOOL_AUTHORIZATION_GRANTS_TABLE}
        (policy_id, tool_name, scope_kind, scope_id, status, created_at_ms, grant_id);
    CREATE INDEX IF NOT EXISTS cayu_tool_authorization_grants_order
        ON {TOOL_AUTHORIZATION_GRANTS_TABLE} (created_at_ms, grant_id);
    CREATE TABLE IF NOT EXISTS {TOOL_AUTHORIZATION_GRANT_USES_TABLE} (
        grant_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        session_instance_id TEXT NOT NULL,
        tool_round_id TEXT NOT NULL,
        tool_call_id TEXT NOT NULL,
        arguments_digest TEXT NOT NULL,
        recorded_at_ms INTEGER NOT NULL,
        record_json TEXT NOT NULL,
        PRIMARY KEY (
            grant_id, session_id, session_instance_id, tool_round_id, tool_call_id, arguments_digest
        )
    );
"""

POSTGRES_TOOL_AUTHORIZATION_GRANT_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {TOOL_AUTHORIZATION_GRANTS_TABLE} (
        grant_id TEXT PRIMARY KEY,
        policy_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'workspace')),
        scope_id TEXT NOT NULL,
        origin_session_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('staged', 'active', 'abandoned', 'revoked')),
        created_at_ms BIGINT NOT NULL,
        record_json TEXT NOT NULL
    )""",
    f"""CREATE INDEX IF NOT EXISTS cayu_tool_authorization_grants_scope
        ON {TOOL_AUTHORIZATION_GRANTS_TABLE}
        (policy_id, tool_name, scope_kind, scope_id, status, created_at_ms, grant_id)""",
    f"""CREATE INDEX IF NOT EXISTS cayu_tool_authorization_grants_order
        ON {TOOL_AUTHORIZATION_GRANTS_TABLE} (created_at_ms, grant_id)""",
    f"""CREATE TABLE IF NOT EXISTS {TOOL_AUTHORIZATION_GRANT_USES_TABLE} (
        grant_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        session_instance_id TEXT NOT NULL,
        tool_round_id TEXT NOT NULL,
        tool_call_id TEXT NOT NULL,
        arguments_digest TEXT NOT NULL,
        recorded_at_ms BIGINT NOT NULL,
        record_json TEXT NOT NULL,
        PRIMARY KEY (
            grant_id, session_id, session_instance_id, tool_round_id, tool_call_id, arguments_digest
        )
    )""",
)


__all__ = [
    "POSTGRES_TOOL_AUTHORIZATION_GRANT_DDL",
    "SQLITE_TOOL_AUTHORIZATION_GRANT_DDL",
    "TOOL_AUTHORIZATION_GRANTS_TABLE",
    "TOOL_AUTHORIZATION_GRANT_REVISION",
    "TOOL_AUTHORIZATION_GRANT_USES_TABLE",
]
