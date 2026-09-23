"""Forward repair and structural checks for durable participant bindings."""

from typing import Any

from cayu.storage.migrations import SchemaError

SQLITE_PARTICIPANT_BINDINGS_DDL = """
        CREATE TABLE IF NOT EXISTS cayu_participant_session_bindings (
            creation_key TEXT PRIMARY KEY,
            request_commitment TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            session_instance_id TEXT NOT NULL,
            application_scope TEXT NOT NULL,
            participant_owner_id TEXT NOT NULL,
            participant_owner_incarnation TEXT NOT NULL,
            participant_id TEXT NOT NULL,
            participant_incarnation TEXT NOT NULL,
            lifecycle_revision INTEGER NOT NULL,
            configuration_revision INTEGER NOT NULL,
            admission_generation INTEGER NOT NULL,
            creator_commitment TEXT NOT NULL,
            authorization_commitment TEXT NOT NULL,
            initial_input_commitment TEXT NOT NULL,
            execution_profile_commitment TEXT NOT NULL,
            binding_json TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            CHECK (length(creation_key) BETWEEN 1 AND 256),
            CHECK (length(request_commitment) BETWEEN 1 AND 256)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_participant_session_bindings_participant
            ON cayu_participant_session_bindings(participant_owner_id, participant_id);
    """

POSTGRES_PARTICIPANT_BINDINGS_DDL = (
    """
        CREATE TABLE IF NOT EXISTS cayu_participant_session_bindings (
            creation_key TEXT PRIMARY KEY,
            request_commitment TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            session_instance_id TEXT NOT NULL,
            application_scope TEXT NOT NULL,
            participant_owner_id TEXT NOT NULL,
            participant_owner_incarnation TEXT NOT NULL,
            participant_id TEXT NOT NULL,
            participant_incarnation TEXT NOT NULL,
            lifecycle_revision BIGINT NOT NULL,
            configuration_revision BIGINT NOT NULL,
            admission_generation BIGINT NOT NULL,
            creator_commitment TEXT NOT NULL,
            authorization_commitment TEXT NOT NULL,
            initial_input_commitment TEXT NOT NULL,
            execution_profile_commitment TEXT NOT NULL,
            binding_json JSONB NOT NULL,
            receipt_json JSONB NOT NULL,
            CHECK (char_length(creation_key) BETWEEN 1 AND 256),
            CHECK (char_length(request_commitment) BETWEEN 1 AND 256)
        )
        """,
    """
        CREATE INDEX IF NOT EXISTS idx_cayu_participant_session_bindings_participant
            ON cayu_participant_session_bindings(participant_owner_id, participant_id)
        """,
)

PARTICIPANT_BINDING_COLUMNS = (
    "creation_key",
    "request_commitment",
    "session_id",
    "session_instance_id",
    "application_scope",
    "participant_owner_id",
    "participant_owner_incarnation",
    "participant_id",
    "participant_incarnation",
    "lifecycle_revision",
    "configuration_revision",
    "admission_generation",
    "creator_commitment",
    "authorization_commitment",
    "initial_input_commitment",
    "execution_profile_commitment",
    "binding_json",
    "receipt_json",
)
PARTICIPANT_BINDING_PROJECTION = ", ".join(PARTICIPANT_BINDING_COLUMNS)
_NUMERIC = {"lifecycle_revision", "configuration_revision", "admission_generation"}
_TABLE = "cayu_participant_session_bindings"
_INDEX = "idx_cayu_participant_session_bindings_participant"
_LOOKUP = ("participant_owner_id", "participant_id")


def _conflict() -> SchemaError:
    return SchemaError(
        "Participant session bindings schema is missing or conflicts with the required "
        "table/index contract. Run `cayu storage migrate` to apply revision 102; "
        "if already applied, restore a known-good schema."
    )


def validate_sqlite_participant_bindings(connection: Any) -> None:
    rows = connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
    expected = {
        name: (
            "INTEGER" if name in _NUMERIC else "TEXT",
            int(name != "creation_key"),
            int(name == "creation_key"),
        )
        for name in PARTICIPANT_BINDING_COLUMNS
    }
    actual = {row[1]: (*row[2:4], row[5]) for row in rows}
    # Additive revisions may extend the table without changing required columns.
    if any(actual.get(name) != definition for name, definition in expected.items()):
        raise _conflict()
    indexes = connection.execute(f"PRAGMA index_list({_TABLE})").fetchall()
    unique = set()
    lookup = False
    for row in indexes:
        # Catalog names are quoted as string literals, never executable SQL.
        name = str(row[1]).replace("'", "''")
        columns = tuple(r[2] for r in connection.execute(f"PRAGMA index_info('{name}')"))
        if row[2] and not row[4]:
            unique.add(columns)
        if row[1] == _INDEX and not row[2] and not row[4] and columns == _LOOKUP:
            lookup = True
    if not lookup or not {("creation_key",), ("session_id",)} <= unique:
        raise _conflict()
    foreign = connection.execute(f"PRAGMA foreign_key_list({_TABLE})").fetchall()
    if not any(
        tuple(row[2:7]) == ("cayu_sessions", "session_id", "id", "NO ACTION", "CASCADE")
        for row in foreign
    ):
        raise _conflict()


async def validate_postgres_participant_bindings(cursor: Any) -> None:
    await cursor.execute(
        """SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
        FROM pg_attribute a WHERE a.attrelid=to_regclass(%s)
        AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum""",
        (_TABLE,),
    )
    expected = {
        name: (
            "bigint"
            if name in _NUMERIC
            else "jsonb"
            if name in {"binding_json", "receipt_json"}
            else "text",
            True,
        )
        for name in PARTICIPANT_BINDING_COLUMNS
    }
    actual = {
        name: (column_type, not_null) for name, column_type, not_null in await cursor.fetchall()
    }
    if any(actual.get(name) != definition for name, definition in expected.items()):
        raise _conflict()
    await cursor.execute(
        """SELECT c.relname, i.indisunique, i.indisprimary,
        array_agg(a.attname ORDER BY k.ordinality)
        FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
        CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality)
        JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
        WHERE i.indrelid=to_regclass(%s) AND i.indisvalid AND i.indisready
        AND i.indpred IS NULL AND i.indexprs IS NULL
        GROUP BY c.relname, i.indisunique, i.indisprimary""",
        (_TABLE,),
    )
    rows = await cursor.fetchall()
    if not any(
        name == _INDEX and not unique and tuple(cols) == _LOOKUP
        for name, unique, primary, cols in rows
    ):
        raise _conflict()
    if not any(primary and cols == ["creation_key"] for _, _, primary, cols in rows):
        raise _conflict()
    if not any(unique and cols == ["session_id"] for _, unique, _, cols in rows):
        raise _conflict()
    await cursor.execute(
        """SELECT pg_get_constraintdef(oid) FROM pg_constraint
        WHERE conrelid=to_regclass(%s) AND contype='f' AND convalidated""",
        (_TABLE,),
    )
    if (
        "FOREIGN KEY (session_id) REFERENCES cayu_sessions(id) ON DELETE CASCADE",
    ) not in await cursor.fetchall():
        raise _conflict()
