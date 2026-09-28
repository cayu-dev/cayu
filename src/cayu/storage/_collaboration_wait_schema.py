"""Source-owned wait discovery projection, never a second wait state machine."""

from typing import Any

from cayu.storage.migrations import SchemaError

_TABLE = "cayu_collaboration_wait_discovery"
_COLUMNS = ("scope", "namespace", "generation", "caller_key", "state", "delivery")
_PRIMARY = _COLUMNS[:4]
_WAIT_INDEX = "idx_cayu_collaboration_wait_discovery_order"
_SESSION_INDEX = "idx_cayu_participant_session_discovery"
_SESSION_COLUMNS = ("participant_owner_id", "participant_id", "creation_key")


def _ddl(postgres: bool) -> tuple[str, ...]:
    mode = "document::jsonb->>'mode'" if postgres else "json_extract(document, '$.mode')"
    state = "document::jsonb->>'state'" if postgres else "json_extract(document, '$.state')"
    delivery = (
        "document::jsonb->>'delivery'" if postgres else "json_extract(document, '$.delivery')"
    )
    statements = (
        f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
            scope TEXT NOT NULL,
            namespace TEXT NOT NULL,
            generation BIGINT NOT NULL,
            caller_key TEXT NOT NULL,
            state TEXT NOT NULL,
            delivery TEXT NOT NULL,
            PRIMARY KEY (scope, namespace, generation, caller_key),
            FOREIGN KEY (scope, namespace, generation, caller_key)
              REFERENCES cayu_collaboration_operations(scope, namespace, generation, caller_key)
        )""",
        # Existing source records remain authoritative during migration. Do not
        # manufacture new wait registrations or depend on a process-local list.
        f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) "
        f"SELECT scope, namespace, generation, caller_key, {state}, {delivery} "
        f"FROM cayu_collaboration_operations WHERE {mode}='collaboration_wait' "
        "ON CONFLICT (scope, namespace, generation, caller_key) DO NOTHING",
        f"CREATE INDEX IF NOT EXISTS {_SESSION_INDEX} "
        "ON cayu_participant_session_bindings(participant_owner_id, participant_id, "
        + ('creation_key COLLATE "C")' if postgres else "creation_key)"),
    )
    if postgres:
        # A locale-dependent primary key cannot provide the canonical ordered
        # range scan. SQLite's existing primary key already uses BINARY order.
        statements += (
            f"CREATE INDEX IF NOT EXISTS {_WAIT_INDEX} ON {_TABLE} "
            '(scope, namespace, generation, caller_key COLLATE "C")',
        )
    return statements


POSTGRES_COLLABORATION_WAIT_DDL = _ddl(True)
SQLITE_COLLABORATION_WAIT_DDL = ";\n".join(_ddl(False)) + ";"


def validate_sqlite_wait_discovery(connection: Any) -> None:
    rows = connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
    expected = [
        (name, "BIGINT" if name == "generation" else "TEXT", 1, index + 1 if index < 4 else 0)
        for index, name in enumerate(_COLUMNS)
    ]
    if [(row[1], row[2], row[3], row[5]) for row in rows] != expected:
        raise SchemaError("Collaboration wait discovery schema conflicts.")
    foreign = connection.execute(f"PRAGMA foreign_key_list({_TABLE})").fetchall()
    if [(row[2], row[3], row[4]) for row in foreign] != [
        ("cayu_collaboration_operations", name, name) for name in _PRIMARY
    ]:
        raise SchemaError("Collaboration wait discovery source linkage conflicts.")
    if (
        tuple(row[2] for row in connection.execute(f"PRAGMA index_info({_SESSION_INDEX})"))
        != _SESSION_COLUMNS
    ):
        raise SchemaError("Participant session discovery index conflicts.")
    if not any(
        row[1] == _SESSION_INDEX and not row[2] and not row[4]
        for row in connection.execute("PRAGMA index_list(cayu_participant_session_bindings)")
    ):
        raise SchemaError("Participant session discovery index is missing or partial.")


async def validate_postgres_wait_discovery(cursor: Any) -> None:
    await cursor.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a WHERE a.attrelid=to_regclass(%s) "
        "AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum",
        (_TABLE,),
    )
    if await cursor.fetchall() != [
        (name, "bigint" if name == "generation" else "text", True) for name in _COLUMNS
    ]:
        raise SchemaError("Collaboration wait discovery schema conflicts.")
    await cursor.execute(
        "SELECT array_agg(a.attname ORDER BY k.ordinality) FROM pg_index i "
        "CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum "
        "WHERE i.indrelid=to_regclass(%s) AND i.indisprimary AND i.indisvalid "
        "GROUP BY i.indexrelid",
        (_TABLE,),
    )
    if await cursor.fetchall() != [(list(_PRIMARY),)]:
        raise SchemaError("Collaboration wait discovery primary key conflicts.")
    await cursor.execute(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid=to_regclass(%s) AND contype='f' AND convalidated",
        (_TABLE,),
    )
    expected = (
        "FOREIGN KEY (scope, namespace, generation, caller_key) REFERENCES "
        "cayu_collaboration_operations(scope, namespace, generation, caller_key)"
    )
    if await cursor.fetchall() != [(expected,)]:
        raise SchemaError("Collaboration wait discovery source linkage conflicts.")
    await _validate_postgres_order_index(cursor, _TABLE, _WAIT_INDEX, _PRIMARY)
    await _validate_postgres_order_index(
        cursor, "cayu_participant_session_bindings", _SESSION_INDEX, _SESSION_COLUMNS
    )


async def _validate_postgres_order_index(cursor, table, index, columns):
    await cursor.execute(
        "SELECT array_agg(a.attname ORDER BY k.ordinality), "
        "array_agg(c.collname ORDER BY k.ordinality) FROM pg_index i "
        "CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum "
        "LEFT JOIN pg_collation c ON c.oid=i.indcollation[k.ordinality - 1] "
        "WHERE i.indexrelid=to_regclass(%s) AND i.indisvalid AND i.indisready AND NOT i.indisunique "
        "AND i.indrelid=to_regclass(%s) "
        "AND i.indpred IS NULL AND i.indexprs IS NULL GROUP BY i.indexrelid",
        (index, table),
    )
    rows = await cursor.fetchall()
    if len(rows) != 1 or rows[0][0] != list(columns) or rows[0][1][-1] != "C":
        raise SchemaError("Collaboration discovery ordering index conflicts.")
