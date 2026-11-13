"""The ``cayu_artifact_index`` tables, shared by the SQLite and Postgres session stores.

``cayu_artifact_index`` holds one row per artifact of each artifact store, written
by the artifact store commit hook (``cayu.artifacts._index``): its metadata, the
columns lists filter and sort on, a lowercased search text, the current title of
the session that saved it, and ``deleted_at`` once it is deleted. Rows are never
removed, so a late write of a deleted artifact cannot bring it back.

``cayu_artifact_index_stores`` records which stores have been filled from the
store itself. Reads use the index only for those stores.

Triggers on ``cayu_sessions`` keep each live row's title current, for every
writer, older binaries included; they touch only live rows, which the partial
session index finds, and a revived row reads the title afresh. Search uses an FTS5 trigram table on SQLite and
``pg_trgm`` indexes on Postgres when the extension can be installed; without it
Postgres searches the narrow table by scanning it.
"""

from __future__ import annotations

#: The storage revision that adds the index. Renumber here when main moves.
ARTIFACT_INDEX_REVISION = 153

_SQLITE_TITLE = """CASE
    WHEN json_type({row}.metadata_json, '$."cayu:title"') = 'text'
    AND length(json_extract({row}.metadata_json, '$."cayu:title"')) <= 256
    THEN json_extract({row}.metadata_json, '$."cayu:title"') END"""


def sqlite_session_title(row: str) -> str:
    """The title ``read_session_titles`` reads, as SQL over a ``cayu_sessions`` row."""

    return _SQLITE_TITLE.format(row=row)


_LIVE_INDEXES = (
    ("created", "created_at"),
    ("session", "session_id, created_at"),
    ("environment", "environment_name, created_at"),
    ("agent", "agent_name, created_at"),
    ("filename", "filename_key"),
    ("size", "size_bytes"),
)


def _index_ddl() -> list[str]:
    return [
        f"""CREATE INDEX IF NOT EXISTS idx_cayu_artifact_index_{name}
        ON cayu_artifact_index({columns}, artifact_store_id, artifact_id)
        WHERE deleted_at IS NULL"""
        for name, columns in _LIVE_INDEXES
    ]


SQLITE_ARTIFACT_INDEX_DDL = (
    """
CREATE TABLE IF NOT EXISTS cayu_artifact_index (
    row_id INTEGER PRIMARY KEY,
    artifact_store_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    scope TEXT,
    session_id TEXT,
    agent_name TEXT,
    environment_name TEXT,
    created_at TEXT,
    filename TEXT,
    filename_key TEXT,
    content_type TEXT,
    size_bytes INTEGER,
    search_text TEXT,
    title TEXT,
    metadata_json TEXT,
    indexed_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE (artifact_store_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS cayu_artifact_index_stores (
    artifact_store_id TEXT PRIMARY KEY,
    backfilled_at TEXT
);
"""
    + ";\n".join(_index_ddl())
    + f""";
CREATE VIRTUAL TABLE IF NOT EXISTS cayu_artifact_index_fts USING fts5(
    search_text, title,
    content='cayu_artifact_index', content_rowid='row_id',
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_fts_insert
AFTER INSERT ON cayu_artifact_index
BEGIN
    INSERT INTO cayu_artifact_index_fts(rowid, search_text, title)
    VALUES (NEW.row_id, NEW.search_text, NEW.title);
END;
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_fts_delete
AFTER DELETE ON cayu_artifact_index
BEGIN
    INSERT INTO cayu_artifact_index_fts(cayu_artifact_index_fts, rowid, search_text, title)
    VALUES ('delete', OLD.row_id, OLD.search_text, OLD.title);
END;
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_fts_update
AFTER UPDATE OF search_text, title ON cayu_artifact_index
BEGIN
    INSERT INTO cayu_artifact_index_fts(cayu_artifact_index_fts, rowid, search_text, title)
    VALUES ('delete', OLD.row_id, OLD.search_text, OLD.title);
    INSERT INTO cayu_artifact_index_fts(rowid, search_text, title)
    VALUES (NEW.row_id, NEW.search_text, NEW.title);
END;
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_title_insert
AFTER INSERT ON cayu_sessions
BEGIN
    UPDATE cayu_artifact_index SET title = {sqlite_session_title("NEW")}
    WHERE session_id = NEW.id AND deleted_at IS NULL
    AND title IS NOT {sqlite_session_title("NEW")};
END;
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_title_update
AFTER UPDATE OF metadata_json ON cayu_sessions
WHEN {sqlite_session_title("OLD")} IS NOT {sqlite_session_title("NEW")}
BEGIN
    UPDATE cayu_artifact_index SET title = {sqlite_session_title("NEW")}
    WHERE session_id = NEW.id AND deleted_at IS NULL;
END;
CREATE TRIGGER IF NOT EXISTS cayu_artifact_index_title_delete
AFTER DELETE ON cayu_sessions
BEGIN
    UPDATE cayu_artifact_index SET title = NULL
    WHERE session_id = OLD.id AND deleted_at IS NULL AND title IS NOT NULL;
END;
"""
)

_POSTGRES_TITLE = """CASE
    WHEN jsonb_typeof({row}.metadata -> 'cayu:title') = 'string'
    AND length({row}.metadata ->> 'cayu:title') <= 256
    THEN {row}.metadata ->> 'cayu:title' END"""


def postgres_session_title(row: str) -> str:
    return _POSTGRES_TITLE.format(row=row)


POSTGRES_ARTIFACT_INDEX_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_artifact_index (
        artifact_store_id TEXT COLLATE "C" NOT NULL,
        artifact_id TEXT COLLATE "C" NOT NULL,
        scope TEXT,
        session_id TEXT,
        agent_name TEXT,
        environment_name TEXT,
        created_at TIMESTAMPTZ,
        filename TEXT,
        filename_key TEXT COLLATE "C",
        content_type TEXT,
        size_bytes BIGINT,
        search_text TEXT,
        title TEXT,
        metadata_json TEXT,
        indexed_at TIMESTAMPTZ NOT NULL,
        deleted_at TIMESTAMPTZ,
        PRIMARY KEY (artifact_store_id, artifact_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_artifact_index_stores (
        artifact_store_id TEXT PRIMARY KEY,
        backfilled_at TIMESTAMPTZ
    )
    """,
    *_index_ddl(),
    # pg_trgm is a trusted extension, but a role may still lack CREATE on the
    # database. Search then scans the narrow table instead of failing the migration.
    """
    DO $$
    DECLARE
        trgm_schema name;
    BEGIN
        BEGIN
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
        EXCEPTION WHEN insufficient_privilege OR undefined_file OR feature_not_supported THEN
            NULL;
        END;
        -- The extension may live in another schema than the store's (it is
        -- installed once per database), so its operator class is named in full.
        SELECT namespace.nspname INTO trgm_schema
        FROM pg_extension AS extension
        JOIN pg_namespace AS namespace ON namespace.oid = extension.extnamespace
        WHERE extension.extname = 'pg_trgm';
        IF trgm_schema IS NOT NULL THEN
            EXECUTE format(
                'CREATE INDEX IF NOT EXISTS idx_cayu_artifact_index_search
                ON cayu_artifact_index USING gin (search_text %I.gin_trgm_ops)
                WHERE deleted_at IS NULL',
                trgm_schema
            );
            EXECUTE format(
                'CREATE INDEX IF NOT EXISTS idx_cayu_artifact_index_title_search
                ON cayu_artifact_index USING gin (lower(title) %I.gin_trgm_ops)
                WHERE deleted_at IS NULL',
                trgm_schema
            );
        END IF;
    END
    $$
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_artifact_index_title()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        IF TG_OP = 'DELETE' THEN
            UPDATE cayu_artifact_index SET title = NULL
            WHERE session_id = OLD.id AND deleted_at IS NULL AND title IS NOT NULL;
        ELSE
            UPDATE cayu_artifact_index SET title = {postgres_session_title("NEW")}
            WHERE session_id = NEW.id AND deleted_at IS NULL
            AND title IS DISTINCT FROM {postgres_session_title("NEW")};
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_artifact_index_title_insert ON cayu_sessions",
    """
    CREATE TRIGGER cayu_artifact_index_title_insert
    AFTER INSERT ON cayu_sessions
    FOR EACH ROW EXECUTE FUNCTION cayu_track_artifact_index_title()
    """,
    "DROP TRIGGER IF EXISTS cayu_artifact_index_title_update ON cayu_sessions",
    f"""
    CREATE TRIGGER cayu_artifact_index_title_update
    AFTER UPDATE OF metadata ON cayu_sessions
    FOR EACH ROW
    WHEN ({postgres_session_title("OLD")} IS DISTINCT FROM {postgres_session_title("NEW")})
    EXECUTE FUNCTION cayu_track_artifact_index_title()
    """,
    "DROP TRIGGER IF EXISTS cayu_artifact_index_title_delete ON cayu_sessions",
    """
    CREATE TRIGGER cayu_artifact_index_title_delete
    AFTER DELETE ON cayu_sessions
    FOR EACH ROW EXECUTE FUNCTION cayu_track_artifact_index_title()
    """,
)
