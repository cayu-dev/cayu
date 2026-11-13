"""Each knowledge entry's current status and list order, kept by triggers.

Knowledge lists filter on an entry's current status and sort on its current
revision's importance and update time. Both live in ``cayu_knowledge_revisions``,
next to every earlier revision, so a status filter scans every revision ever
written in that status (each kept entry leaves its pending proposal behind) and
the sort reads each current revision. ``cayu_knowledge_entry_order`` keeps one
row per entry with its namespace, current status and sort key, written by
triggers on the entry and revision tables in the same transaction, for every
writer. Lists and counts drive from its ``(status, sort key)`` index and join
the current entry for the remaining filters.

A row exists once an entry and its current revision both exist; the revision
may be written later in the same transaction (its foreign key is deferred).
"""

from __future__ import annotations

from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeListQuery

#: The list order: importance (missing as 0.0) descending, then update time
#: descending, then entry ID ascending, as ``encode_knowledge_list_cursor`` names.
ORDER_BY_SQL = "listed.sort_importance DESC, listed.sort_updated_at DESC, listed.entry_id ASC"


def _sqlite_refresh(entry_id: str) -> str:
    return f"""
    DELETE FROM cayu_knowledge_entry_order WHERE entry_id = {entry_id};
    INSERT INTO cayu_knowledge_entry_order (
        entry_id, namespace, status, sort_importance, sort_updated_at, expires_at
    )
    SELECT logical.id, logical.namespace, revision.status,
           COALESCE(revision.importance, 0.0), revision.updated_at, revision.expires_at
    FROM cayu_knowledge_entries AS logical
    JOIN cayu_knowledge_revisions AS revision
      ON revision.entry_id = logical.id AND revision.revision = logical.current_revision
    WHERE logical.id = {entry_id};"""


SQLITE_KNOWLEDGE_ORDER_DDL = f"""
CREATE TABLE IF NOT EXISTS cayu_knowledge_entry_order (
    entry_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    status TEXT NOT NULL,
    sort_importance REAL NOT NULL,
    sort_updated_at TEXT NOT NULL,
    expires_at TEXT
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entry_order_status
    ON cayu_knowledge_entry_order(
        status, sort_importance DESC, sort_updated_at DESC, entry_id, namespace, expires_at
    );

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_entry_order_entry_insert
AFTER INSERT ON cayu_knowledge_entries
FOR EACH ROW
BEGIN
    {_sqlite_refresh("NEW.id")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_entry_order_entry_update
AFTER UPDATE OF id, namespace, current_revision ON cayu_knowledge_entries
FOR EACH ROW
BEGIN
    DELETE FROM cayu_knowledge_entry_order WHERE entry_id = OLD.id;
    {_sqlite_refresh("NEW.id")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_entry_order_entry_delete
AFTER DELETE ON cayu_knowledge_entries
FOR EACH ROW
BEGIN
    DELETE FROM cayu_knowledge_entry_order WHERE entry_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_entry_order_revision_insert
AFTER INSERT ON cayu_knowledge_revisions
FOR EACH ROW
BEGIN
    {_sqlite_refresh("NEW.entry_id")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_entry_order_revision_update
AFTER UPDATE OF entry_id, revision, status, importance, updated_at, expires_at
ON cayu_knowledge_revisions
FOR EACH ROW
BEGIN
    {_sqlite_refresh("OLD.entry_id")}
    {_sqlite_refresh("NEW.entry_id")}
END;

INSERT OR REPLACE INTO cayu_knowledge_entry_order (
    entry_id, namespace, status, sort_importance, sort_updated_at, expires_at
)
SELECT logical.id, logical.namespace, revision.status,
       COALESCE(revision.importance, 0.0), revision.updated_at, revision.expires_at
FROM cayu_knowledge_entries AS logical
JOIN cayu_knowledge_revisions AS revision
  ON revision.entry_id = logical.id AND revision.revision = logical.current_revision;
"""
"""SQLite: the table, its index, the triggers and the backfill."""


POSTGRES_KNOWLEDGE_ORDER_DDL: tuple[str, ...] = (
    "LOCK TABLE cayu_knowledge_entries IN SHARE MODE",
    "LOCK TABLE cayu_knowledge_revisions IN SHARE MODE",
    """
    CREATE TABLE IF NOT EXISTS cayu_knowledge_entry_order (
        entry_id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        status TEXT NOT NULL,
        sort_importance DOUBLE PRECISION NOT NULL,
        sort_updated_at TIMESTAMPTZ NOT NULL,
        expires_at TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entry_order_status
        ON cayu_knowledge_entry_order(
            status, sort_importance DESC, sort_updated_at DESC, entry_id, namespace, expires_at
        )
    """,
    """
    CREATE OR REPLACE FUNCTION cayu_refresh_knowledge_entry_order(refreshed TEXT)
    RETURNS VOID
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        DELETE FROM cayu_knowledge_entry_order WHERE entry_id = refreshed;
        INSERT INTO cayu_knowledge_entry_order (
            entry_id, namespace, status, sort_importance, sort_updated_at, expires_at
        )
        SELECT logical.id, logical.namespace, revision.status,
               COALESCE(revision.importance, 0.0), revision.updated_at, revision.expires_at
        FROM cayu_knowledge_entries AS logical
        JOIN cayu_knowledge_revisions AS revision
          ON revision.entry_id = logical.id
         AND revision.revision = logical.current_revision
        WHERE logical.id = refreshed;
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    CREATE OR REPLACE FUNCTION cayu_track_knowledge_entry_order()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        IF TG_TABLE_NAME = 'cayu_knowledge_entries' THEN
            IF TG_OP IN ('UPDATE', 'DELETE') THEN
                DELETE FROM cayu_knowledge_entry_order WHERE entry_id = OLD.id;
            END IF;
            IF TG_OP IN ('INSERT', 'UPDATE') THEN
                PERFORM cayu_refresh_knowledge_entry_order(NEW.id);
            END IF;
        ELSE
            IF TG_OP = 'UPDATE' THEN
                PERFORM cayu_refresh_knowledge_entry_order(OLD.entry_id);
            END IF;
            PERFORM cayu_refresh_knowledge_entry_order(NEW.entry_id);
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_entry_order_entry ON cayu_knowledge_entries",
    """
    CREATE TRIGGER cayu_knowledge_entry_order_entry
    AFTER INSERT OR DELETE OR UPDATE OF id, namespace, current_revision
    ON cayu_knowledge_entries
    FOR EACH ROW
    EXECUTE FUNCTION cayu_track_knowledge_entry_order()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_entry_order_revision ON cayu_knowledge_revisions",
    """
    CREATE TRIGGER cayu_knowledge_entry_order_revision
    AFTER INSERT OR UPDATE OF entry_id, revision, status, importance, updated_at, expires_at
    ON cayu_knowledge_revisions
    FOR EACH ROW
    EXECUTE FUNCTION cayu_track_knowledge_entry_order()
    """,
    """
    INSERT INTO cayu_knowledge_entry_order (
        entry_id, namespace, status, sort_importance, sort_updated_at, expires_at
    )
    SELECT logical.id, logical.namespace, revision.status,
           COALESCE(revision.importance, 0.0), revision.updated_at, revision.expires_at
    FROM cayu_knowledge_entries AS logical
    JOIN cayu_knowledge_revisions AS revision
      ON revision.entry_id = logical.id AND revision.revision = logical.current_revision
    ON CONFLICT (entry_id) DO UPDATE SET
        namespace = EXCLUDED.namespace,
        status = EXCLUDED.status,
        sort_importance = EXCLUDED.sort_importance,
        sort_updated_at = EXCLUDED.sort_updated_at,
        expires_at = EXCLUDED.expires_at
    """,
)
"""PostgreSQL: the same, with writers paused while the backfill copies."""


def listed_entries_sql(
    *,
    statuses: list[str],
    namespace: str | None,
    cursor: tuple[float, object, str] | None,
    placeholder: str,
) -> tuple[str, str, list[object]]:
    """The ``FROM`` clause and leading ``WHERE`` conditions for a list or count.

    The order table drives (``CROSS JOIN`` keeps that order in SQLite) and the
    current entry, ``e``, is joined by key for every other filter. ``cursor`` is
    a decoded list cursor with its time already in the store's format.
    """

    conditions, params = _listed_conditions(statuses, namespace, cursor, placeholder)
    from_sql = "cayu_knowledge_entry_order AS listed CROSS JOIN cayu_knowledge_current_entries AS e"
    return from_sql, " AND ".join(["e.id = listed.entry_id", *conditions]), params


def order_alone_count_sql(
    query: KnowledgeListQuery,
    scope: KnowledgeAccessScope,
    *,
    cursor: tuple[float, object, str] | None,
    now: object,
    placeholder: str,
) -> tuple[str, list[object]] | None:
    """A count over the order table alone, when nothing else filters the list.

    That holds for an unrestricted scope (``KnowledgeAccessScope.privileged()``,
    as a host's review workflow uses) and a query filtering only by status,
    namespace and expiry, as the inbox's pending count does. ``None`` otherwise:
    the caller then counts through the current entries.
    """

    if (
        scope != KnowledgeAccessScope.privileged()
        or query.labels
        or query.kinds is not None
        or query.visibilities is not None
        or query.aspects
        or query.impact_targets
        or query.source_type is not None
        or query.source_id is not None
    ):
        return None
    conditions, params = _listed_conditions(
        [str(status) for status in query.statuses], query.namespace, cursor, placeholder
    )
    if not query.include_expired:
        conditions.append(f"(listed.expires_at IS NULL OR listed.expires_at > {placeholder})")
        params.append(now)
    where_sql = " AND ".join(conditions)
    return f"SELECT COUNT(*) FROM cayu_knowledge_entry_order AS listed WHERE {where_sql}", params


def _listed_conditions(
    statuses: list[str],
    namespace: str | None,
    cursor: tuple[float, object, str] | None,
    placeholder: str,
) -> tuple[list[str], list[object]]:
    params: list[object] = list(statuses)
    conditions = [f"listed.status IN ({', '.join(placeholder for _ in statuses)})"]
    if namespace is not None:
        conditions.append(f"listed.namespace = {placeholder}")
        params.append(namespace)
    if cursor is not None:
        importance, updated_at, entry_id = cursor
        conditions.append(
            f"""(listed.sort_importance < {placeholder}
                 OR (listed.sort_importance = {placeholder}
                     AND listed.sort_updated_at < {placeholder})
                 OR (listed.sort_importance = {placeholder}
                     AND listed.sort_updated_at = {placeholder}
                     AND listed.entry_id > {placeholder}))"""
        )
        params.extend([importance, importance, updated_at, importance, updated_at, entry_id])
    return conditions, params
