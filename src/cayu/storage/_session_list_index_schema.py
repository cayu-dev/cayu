"""Indexes that let session lists sort and page by keyset without a scan.

Session lists sort by ``updated_at``, ``created_at`` or ``last_activity_at``
descending with ``id`` as the tie-breaker, and the dashboard narrows them by
agent, environment or parent. Without these indexes every list visited every
matching session to sort it. On PostgreSQL they are built ``CONCURRENTLY`` so
session writes stay available during the upgrade.
"""

from __future__ import annotations

#: The storage revision that adds these indexes. Several revisions are in
#: flight at once; this is the one number to change when renumbering.
SESSION_LIST_INDEX_REVISION = 158

#: (index name, key columns as SQL, key columns as the catalog reports them).
SESSION_LIST_INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("idx_cayu_sessions_updated_id", "updated_at DESC, id", ("updated_at", "id")),
    ("idx_cayu_sessions_created_id", "created_at DESC, id", ("created_at", "id")),
    (
        "idx_cayu_sessions_activity_id",
        "last_activity_at DESC, id",
        ("last_activity_at", "id"),
    ),
    (
        "idx_cayu_sessions_agent_updated_id",
        "agent_name, updated_at DESC, id",
        ("agent_name", "updated_at", "id"),
    ),
    (
        "idx_cayu_sessions_environment_updated_id",
        "environment_name, updated_at DESC, id",
        ("environment_name", "updated_at", "id"),
    ),
    (
        "idx_cayu_sessions_parent_updated_id",
        "parent_session_id, updated_at DESC, id",
        ("parent_session_id", "updated_at", "id"),
    ),
)

SQLITE_SESSION_LIST_INDEX_DDL = "\n".join(
    f"CREATE INDEX IF NOT EXISTS {name} ON cayu_sessions({keys});"
    for name, keys, _ in SESSION_LIST_INDEXES
)

__all__ = [
    "SESSION_LIST_INDEXES",
    "SESSION_LIST_INDEX_REVISION",
    "SQLITE_SESSION_LIST_INDEX_DDL",
]
