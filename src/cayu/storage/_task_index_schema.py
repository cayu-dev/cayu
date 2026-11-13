"""Task list indexes, status counts and prerequisite edges, kept in step by triggers.

The task list, its status chips and a task's dependents read these instead of
scanning and decoding every task:

- ``cayu_tasks.is_internal`` is generated from the task type, so every writer,
  older binaries included, gets it right.
- List indexes cover the filters and the ``(updated_at DESC, id)`` keyset order,
  and carry the other filter columns, so a filtered page checks them in the index
  instead of reading each task's row. On SQLite a covering search index also
  carries the search columns and ``search_folds`` (whether a stored value holds a
  character that casefolds to ASCII), so a text search reads no task rows.
- ``cayu_task_counts`` holds task counts by internal flag, status, type and
  assigned agent (``''`` for none). SQLite has one writer, so its triggers update
  the counts directly. PostgreSQL writers would contend for, and could deadlock
  on, shared count rows, so its triggers append to ``cayu_task_count_changes``
  and readers fold those changes into ``cayu_task_counts`` (see
  ``POSTGRES_FOLD_TASK_COUNT_CHANGES``). A count is always the counts row plus
  its pending changes.
- ``cayu_task_prerequisites`` holds one ``(task_id, prerequisite_task_id)`` edge
  per prerequisite, indexed both ways.

The revision copies existing tasks into the counts and edges. ``cayu task counts
--recompute`` compares the counts with a recount of ``cayu_tasks``.
"""

from __future__ import annotations

#: The storage revision that installs everything in this module. Several changes
#: add revisions at once; renumber here (and in ``migrations.REVISIONS``) only.
TASK_INDEX_REVISION = 151


def _sqlite_internal(row: str) -> str:
    return f"(substr({row}.type, 1, 5) = 'cayu.')"


def _sqlite_key(row: str) -> str:
    return (
        f"{_sqlite_internal(row)}, {row}.status, {row}.type, "
        f"coalesce({row}.assigned_agent_name, '')"
    )


def _sqlite_key_match(row: str) -> str:
    return (
        f"is_internal = {_sqlite_internal(row)} AND status = {row}.status "
        f"AND type = {row}.type AND assigned_agent_name = coalesce({row}.assigned_agent_name, '')"
    )


def _sqlite_count(row: str, delta: str) -> str:
    if delta == "+1":
        return f"""INSERT INTO cayu_task_counts (
            is_internal, status, type, assigned_agent_name, task_count
        ) VALUES ({_sqlite_key(row)}, 1)
        ON CONFLICT (is_internal, status, type, assigned_agent_name)
        DO UPDATE SET task_count = task_count + 1;"""
    return f"""UPDATE cayu_task_counts SET task_count = task_count - 1
        WHERE {_sqlite_key_match(row)};
        DELETE FROM cayu_task_counts WHERE {_sqlite_key_match(row)} AND task_count = 0;"""


def _sqlite_edges(row: str) -> str:
    return f"""INSERT OR IGNORE INTO cayu_task_prerequisites (task_id, prerequisite_task_id)
        SELECT {row}.id, edge.value FROM json_each(
            CASE WHEN json_type({row}.prerequisite_task_ids_json) = 'array'
            THEN {row}.prerequisite_task_ids_json ELSE '[]' END
        ) AS edge
        WHERE edge.type = 'text';"""


#: Characters outside ASCII whose casefold contains ASCII (``ß`` is ``ss``). A
#: stored value holding one can match an ASCII search needle that its plain text
#: does not. ``tests/core/test_task_index_projection.py`` recomputes this set.
CASEFOLD_TO_ASCII = "".join(
    chr(code)
    for code in (
        0x00DF, 0x0130, 0x0149, 0x017F, 0x01F0, 0x1E96, 0x1E97, 0x1E98, 0x1E99, 0x1E9A,
        0x1E9E, 0x212A, 0xFB00, 0xFB01, 0xFB02, 0xFB03, 0xFB04, 0xFB05, 0xFB06,
    )
)  # fmt: skip

#: The free-text task fields a stored value of could casefold to ASCII. Identities,
#: statuses and the other search fields are checked the same way.
_SEARCH_TEXT_COLUMNS = (
    "type",
    "title",
    "description",
    "session_id",
    "parent_task_id",
    "assigned_agent_name",
    "worker_id",
    "status_reason",
    "id",
)

_SQLITE_SEARCH_FOLDS = " OR ".join(
    f"coalesce({column}, '') GLOB '*[{CASEFOLD_TO_ASCII}]*'" for column in _SEARCH_TEXT_COLUMNS
)

#: ``(table, column, declaration)`` added before the revision's DDL.
SQLITE_TASK_INDEX_COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "cayu_tasks",
        "is_internal",
        "INTEGER GENERATED ALWAYS AS (substr(type, 1, 5) = 'cayu.') VIRTUAL",
    ),
    (
        "cayu_tasks",
        "search_folds",
        f"INTEGER GENERATED ALWAYS AS ({_SQLITE_SEARCH_FOLDS}) VIRTUAL",
    ),
)

#: The search fields joined by ``\x1f``, with an immutable expression PostgreSQL
#: can store (``concat_ws`` is only stable).
POSTGRES_SEARCH_TEXT = (
    "("
    + " || E'\\x1f' || ".join(
        f"coalesce({column}, '')"
        for column in (
            "id",
            "type",
            "title",
            "description",
            "status",
            "session_id",
            "parent_task_id",
            "assigned_agent_name",
            "worker_id",
            "status_reason",
        )
    )
    + ")"
)

#: The SQLite index a text search scans; it holds every column the scan reads.
SQLITE_TASK_SEARCH_INDEX = "idx_cayu_tasks_list_search"

SQLITE_TASK_INDEX_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_updated
    ON cayu_tasks(updated_at DESC, id);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_status_updated
    ON cayu_tasks(status, updated_at DESC, id, is_internal, type, assigned_agent_name);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_internal_updated
    ON cayu_tasks(is_internal, updated_at DESC, id, status, type, assigned_agent_name);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_internal_status_updated
    ON cayu_tasks(is_internal, status, updated_at DESC, id, type, assigned_agent_name);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_type_updated
    ON cayu_tasks(type, updated_at DESC, id, is_internal, status, assigned_agent_name);
CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_agent_updated
    ON cayu_tasks(assigned_agent_name, updated_at DESC, id, is_internal, status, type);
CREATE INDEX IF NOT EXISTS {SQLITE_TASK_SEARCH_INDEX}
    ON cayu_tasks(
        updated_at DESC, id, created_at, is_internal, status, type, assigned_agent_name,
        session_id, parent_task_id, worker_id, status_reason, search_folds, title, description
    );

CREATE TABLE IF NOT EXISTS cayu_task_counts (
    is_internal INTEGER NOT NULL,
    status TEXT NOT NULL,
    type TEXT NOT NULL,
    assigned_agent_name TEXT NOT NULL,
    task_count INTEGER NOT NULL,
    PRIMARY KEY (is_internal, status, type, assigned_agent_name)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS cayu_task_prerequisites (
    task_id TEXT NOT NULL,
    prerequisite_task_id TEXT NOT NULL,
    PRIMARY KEY (task_id, prerequisite_task_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_cayu_task_prerequisites_dependents
    ON cayu_task_prerequisites(prerequisite_task_id, task_id);

CREATE TRIGGER IF NOT EXISTS cayu_task_index_insert
AFTER INSERT ON cayu_tasks
FOR EACH ROW
BEGIN
    {_sqlite_count("NEW", "+1")}
    {_sqlite_edges("NEW")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_task_index_delete
AFTER DELETE ON cayu_tasks
FOR EACH ROW
BEGIN
    {_sqlite_count("OLD", "-1")}
    DELETE FROM cayu_task_prerequisites WHERE task_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS cayu_task_index_update_counts
AFTER UPDATE OF status, type, assigned_agent_name ON cayu_tasks
FOR EACH ROW
WHEN OLD.status IS NOT NEW.status OR OLD.type IS NOT NEW.type
    OR OLD.assigned_agent_name IS NOT NEW.assigned_agent_name
BEGIN
    {_sqlite_count("OLD", "-1")}
    {_sqlite_count("NEW", "+1")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_task_index_update_prerequisites
AFTER UPDATE OF id, prerequisite_task_ids_json ON cayu_tasks
FOR EACH ROW
WHEN OLD.id IS NOT NEW.id OR OLD.prerequisite_task_ids_json IS NOT NEW.prerequisite_task_ids_json
BEGIN
    DELETE FROM cayu_task_prerequisites WHERE task_id = OLD.id;
    {_sqlite_edges("NEW")}
END;

DELETE FROM cayu_task_counts;
INSERT INTO cayu_task_counts (is_internal, status, type, assigned_agent_name, task_count)
    SELECT {_sqlite_key("task")}, count(*) FROM cayu_tasks AS task
    GROUP BY 1, 2, 3, 4;
DELETE FROM cayu_task_prerequisites;
INSERT OR IGNORE INTO cayu_task_prerequisites (task_id, prerequisite_task_id)
    SELECT task.id, edge.value FROM cayu_tasks AS task, json_each(
        CASE WHEN json_type(task.prerequisite_task_ids_json) = 'array'
        THEN task.prerequisite_task_ids_json ELSE '[]' END
    ) AS edge
    WHERE task.prerequisite_task_ids_json != '[]' AND edge.type = 'text';
"""


def _postgres_key(row: str) -> str:
    return (
        f"left({row}.type, 5) = 'cayu.', {row}.status, {row}.type, "
        f"coalesce({row}.assigned_agent_name, '')"
    )


def _postgres_change(row: str, delta: int) -> str:
    return f"""INSERT INTO cayu_task_count_changes (
                is_internal, status, type, assigned_agent_name, delta
            ) VALUES ({_postgres_key(row)}, {delta});"""


def _postgres_edges(row: str) -> str:
    return f"""INSERT INTO cayu_task_prerequisites (task_id, prerequisite_task_id)
            SELECT {row}.id, edge.value FROM jsonb_array_elements_text(
                CASE WHEN jsonb_typeof({row}.prerequisite_task_ids) = 'array'
                THEN {row}.prerequisite_task_ids ELSE '[]'::jsonb END
            ) AS edge(value)
            ON CONFLICT DO NOTHING;"""


_POSTGRES_COUNT_KEY = (
    "OLD.status IS DISTINCT FROM NEW.status OR OLD.type IS DISTINCT FROM NEW.type "
    "OR OLD.assigned_agent_name IS DISTINCT FROM NEW.assigned_agent_name"
)
_POSTGRES_EDGE_KEY = (
    "OLD.id IS DISTINCT FROM NEW.id "
    "OR OLD.prerequisite_task_ids IS DISTINCT FROM NEW.prerequisite_task_ids"
)

POSTGRES_TASK_INDEX_DDL: tuple[str, ...] = (
    # One statement, so the table is rewritten once.
    "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS is_internal BOOLEAN "
    "GENERATED ALWAYS AS (left(type, 5) = 'cayu.') STORED, "
    "ADD COLUMN IF NOT EXISTS search_folds BOOLEAN "
    f"GENERATED ALWAYS AS ({POSTGRES_SEARCH_TEXT} ~ '[{CASEFOLD_TO_ASCII}]') STORED",
    'CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_updated ON cayu_tasks(updated_at DESC, id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_status_updated "
    'ON cayu_tasks(status, updated_at DESC, id COLLATE "C", is_internal, type, assigned_agent_name)',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_internal_updated "
    'ON cayu_tasks(is_internal, updated_at DESC, id COLLATE "C", status, type, assigned_agent_name)',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_internal_status_updated "
    'ON cayu_tasks(is_internal, status, updated_at DESC, id COLLATE "C", type, assigned_agent_name)',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_type_updated "
    'ON cayu_tasks(type, updated_at DESC, id COLLATE "C", is_internal, status, assigned_agent_name)',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_list_agent_updated "
    'ON cayu_tasks(assigned_agent_name, updated_at DESC, id COLLATE "C", is_internal, status, type)',
    """CREATE TABLE IF NOT EXISTS cayu_task_counts (
        is_internal BOOLEAN NOT NULL,
        status TEXT NOT NULL,
        type TEXT NOT NULL,
        assigned_agent_name TEXT NOT NULL,
        task_count BIGINT NOT NULL,
        PRIMARY KEY (is_internal, status, type, assigned_agent_name)
    )""",
    """CREATE TABLE IF NOT EXISTS cayu_task_count_changes (
        change_id BIGSERIAL PRIMARY KEY,
        is_internal BOOLEAN NOT NULL,
        status TEXT NOT NULL,
        type TEXT NOT NULL,
        assigned_agent_name TEXT NOT NULL,
        delta INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS cayu_task_prerequisites (
        task_id TEXT NOT NULL,
        prerequisite_task_id TEXT NOT NULL,
        PRIMARY KEY (task_id, prerequisite_task_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cayu_task_prerequisites_dependents "
    "ON cayu_task_prerequisites(prerequisite_task_id, task_id)",
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_task_index()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'DELETE' OR (TG_OP = 'UPDATE' AND ({_POSTGRES_COUNT_KEY})) THEN
            {_postgres_change("OLD", -1)}
        END IF;
        IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND ({_POSTGRES_COUNT_KEY})) THEN
            {_postgres_change("NEW", 1)}
        END IF;
        IF TG_OP = 'DELETE' OR (TG_OP = 'UPDATE' AND ({_POSTGRES_EDGE_KEY})) THEN
            DELETE FROM cayu_task_prerequisites WHERE task_id = OLD.id;
        END IF;
        IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND ({_POSTGRES_EDGE_KEY})) THEN
            {_postgres_edges("NEW")}
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_task_index_insert_delete ON cayu_tasks",
    """CREATE TRIGGER cayu_task_index_insert_delete
    AFTER INSERT OR DELETE ON cayu_tasks
    FOR EACH ROW EXECUTE FUNCTION cayu_track_task_index()""",
    "DROP TRIGGER IF EXISTS cayu_task_index_update ON cayu_tasks",
    """CREATE TRIGGER cayu_task_index_update
    AFTER UPDATE OF id, status, type, assigned_agent_name, prerequisite_task_ids ON cayu_tasks
    FOR EACH ROW EXECUTE FUNCTION cayu_track_task_index()""",
    "DELETE FROM cayu_task_count_changes",
    "DELETE FROM cayu_task_counts",
    f"""INSERT INTO cayu_task_counts (is_internal, status, type, assigned_agent_name, task_count)
    SELECT {_postgres_key("task")}, count(*) FROM cayu_tasks AS task GROUP BY 1, 2, 3, 4""",
    "DELETE FROM cayu_task_prerequisites",
    """INSERT INTO cayu_task_prerequisites (task_id, prerequisite_task_id)
    SELECT task.id, edge.value FROM cayu_tasks AS task
    CROSS JOIN LATERAL jsonb_array_elements_text(
        CASE WHEN jsonb_typeof(task.prerequisite_task_ids) = 'array'
        THEN task.prerequisite_task_ids ELSE '[]'::jsonb END
    ) AS edge(value)
    WHERE task.prerequisite_task_ids <> '[]'::jsonb
    ON CONFLICT DO NOTHING""",
)

#: Serializes folders so two never update the same count rows in different orders.
_POSTGRES_FOLD_LOCK = 0x63617975_74636E74  # "cayutcnt"

#: Folds pending count changes into ``cayu_task_counts`` in one transaction. Only
#: folders take count-row locks, and they hold the advisory lock while they do.
POSTGRES_FOLD_TASK_COUNT_CHANGES: tuple[str, ...] = (
    f"SELECT pg_try_advisory_xact_lock({_POSTGRES_FOLD_LOCK})",
    """WITH moved AS (
        DELETE FROM cayu_task_count_changes
        RETURNING is_internal, status, type, assigned_agent_name, delta
    ), summed AS (
        SELECT is_internal, status, type, assigned_agent_name, sum(delta) AS delta
        FROM moved GROUP BY is_internal, status, type, assigned_agent_name
    )
    INSERT INTO cayu_task_counts AS counts (
        is_internal, status, type, assigned_agent_name, task_count
    )
    SELECT is_internal, status, type, assigned_agent_name, delta FROM summed
    ON CONFLICT (is_internal, status, type, assigned_agent_name)
    DO UPDATE SET task_count = counts.task_count + EXCLUDED.task_count""",
    "DELETE FROM cayu_task_counts WHERE task_count = 0",
)
