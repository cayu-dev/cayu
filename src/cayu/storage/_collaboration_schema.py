"""Normalized identity records share the application's existing schema lifecycle."""

from typing import Any

from cayu.storage.migrations import SchemaError

KEYS = {
    "anchors": (),
    "participants": ("participant_id",),
    "configurations": ("participant_id", "revision"),
    "aliases": ("alias",),
    "operations": ("namespace", "generation", "caller_key"),
    "events": ("sequence",),
    "namespaces": ("namespace", "generation"),
    "lifecycle_history": ("participant_id", "revision"),
    "participant_permits": ("participant_id",),
    "permits": ("namespace", "generation", "caller_key"),
    "requests": ("namespace", "generation", "caller_key"),
    "request_events": ("sequence",),
}
_REQUEST_TABLES = frozenset({"requests", "request_events"})
_REQUEST_INDEXES = {
    "cayu_collaboration_request_recipient_idx": (
        "cayu_collaboration_requests",
        ("scope", "participant_id", "state", "position"),
        False,
    ),
    "cayu_collaboration_request_due_idx": (
        "cayu_collaboration_requests",
        ("scope", "state", "next_due_at_ms", "position"),
        False,
    ),
}
_LIFECYCLE_TABLES = frozenset({"namespaces", "lifecycle_history", "participant_permits", "permits"})
EXTRA_COLUMNS = {
    "permits": ("participant_id", "position", "state"),
    "requests": ("participant_id", "position", "state", "next_due_at_ms"),
}
_NUMERIC_COLUMNS = frozenset({"generation", "revision", "sequence", "position", "next_due_at_ms"})
_LIFECYCLE_INDEXES = {
    "cayu_collaboration_permit_position_idx": (
        "cayu_collaboration_permits",
        ("scope", "participant_id", "position"),
        True,
    ),
    "cayu_collaboration_permit_pending_idx": (
        "cayu_collaboration_permits",
        ("scope", "participant_id", "state", "position"),
        False,
    ),
    "cayu_collaboration_history_operation_idx": (
        "cayu_collaboration_history_uses",
        ("scope", "namespace", "generation", "caller_key"),
        False,
    ),
}


def _record_ddl(family: str, columns: tuple[str, ...], postgres: bool) -> str:
    fields = ["scope TEXT NOT NULL"]
    for column in (*columns, *EXTRA_COLUMNS.get(family, ())):
        sql_type = "BIGINT" if column in _NUMERIC_COLUMNS else "TEXT"
        fields.append(f"{column} {sql_type} NOT NULL")
    check = "jsonb_typeof(document::jsonb) = 'object'" if postgres else "json_valid(document)"
    fields += [
        f"document TEXT NOT NULL CHECK ({check})",
        f"PRIMARY KEY ({', '.join(('scope', *columns))})",
    ]
    return f"CREATE TABLE IF NOT EXISTS cayu_collaboration_{family} ({', '.join(fields)})"


def _ddl(postgres: bool) -> tuple[str, ...]:
    statements = []
    for family, columns in KEYS.items():
        if family in _LIFECYCLE_TABLES or family in _REQUEST_TABLES:
            continue
        statements.append(_record_ddl(family, columns, postgres))
    statements.append("""CREATE TABLE IF NOT EXISTS cayu_collaboration_event_participants (
        scope TEXT NOT NULL, sequence BIGINT NOT NULL, participant_id TEXT NOT NULL,
        PRIMARY KEY (scope, sequence, participant_id),
        FOREIGN KEY (scope, sequence) REFERENCES cayu_collaboration_events(scope, sequence)
    )""")
    statements.append(
        "CREATE INDEX IF NOT EXISTS cayu_collaboration_event_participant_idx ON cayu_collaboration_event_participants(scope, participant_id, sequence)"
    )
    return tuple(statements)


POSTGRES_COLLABORATION_DDL = _ddl(True)
SQLITE_COLLABORATION_DDL = ";\n".join(_ddl(False)) + ";"


def _lifecycle_ddl(postgres: bool) -> tuple[str, ...]:
    return (
        *(
            _record_ddl(family, columns, postgres)
            for family, columns in KEYS.items()
            if family in _LIFECYCLE_TABLES
        ),
        """CREATE TABLE IF NOT EXISTS cayu_collaboration_history_uses (
            scope TEXT NOT NULL, family TEXT NOT NULL, participant_id TEXT NOT NULL,
            revision BIGINT NOT NULL, namespace TEXT NOT NULL, generation BIGINT NOT NULL,
            caller_key TEXT NOT NULL,
            PRIMARY KEY (scope, family, participant_id, revision, namespace, generation, caller_key),
            FOREIGN KEY (scope, namespace, generation, caller_key)
                REFERENCES cayu_collaboration_operations(scope, namespace, generation, caller_key)
        )""",
        *(
            f"CREATE {'UNIQUE ' if unique else ''}INDEX IF NOT EXISTS {name} ON {table} ({', '.join(columns)})"
            for name, (table, columns, unique) in _LIFECYCLE_INDEXES.items()
        ),
    )


POSTGRES_COLLABORATION_LIFECYCLE_DDL = _lifecycle_ddl(True)
SQLITE_COLLABORATION_LIFECYCLE_DDL = ";\n".join(_lifecycle_ddl(False)) + ";"


def _request_ddl(postgres: bool) -> tuple[str, ...]:
    return (
        *(_record_ddl(family, KEYS[family], postgres) for family in sorted(_REQUEST_TABLES)),
        "CREATE INDEX IF NOT EXISTS cayu_collaboration_request_recipient_idx ON cayu_collaboration_requests(scope, participant_id, state, position)",
        "CREATE INDEX IF NOT EXISTS cayu_collaboration_request_due_idx ON cayu_collaboration_requests(scope, state, next_due_at_ms, position)",
    )


POSTGRES_COLLABORATION_REQUEST_DDL = _request_ddl(True)
SQLITE_COLLABORATION_REQUEST_DDL = ";\n".join(_request_ddl(False)) + ";"


def _tables(*, lifecycle: bool, requests: bool = False):
    for family, columns in KEYS.items():
        if family in _LIFECYCLE_TABLES and not lifecycle:
            continue
        if family in _REQUEST_TABLES and not requests:
            continue
        primary = ("scope", *columns)
        yield (
            f"cayu_collaboration_{family}",
            (*primary, *EXTRA_COLUMNS.get(family, ()), "document"),
            primary,
        )
    primary = ("scope", "sequence", "participant_id")
    yield "cayu_collaboration_event_participants", primary, primary
    if lifecycle:
        primary = (
            "scope",
            "family",
            "participant_id",
            "revision",
            "namespace",
            "generation",
            "caller_key",
        )
        yield "cayu_collaboration_history_uses", primary, primary


def validate_sqlite_collaboration_schema(
    connection: Any, *, lifecycle: bool = False, requests: bool = False
) -> None:
    for table, columns, primary in _tables(lifecycle=lifecycle, requests=requests):
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        expected = [
            (
                name,
                "BIGINT" if name in _NUMERIC_COLUMNS else "TEXT",
                1,
                primary.index(name) + 1 if name in primary else 0,
            )
            for name in columns
        ]
        if [(row[1], row[2], row[3], row[5]) for row in rows] != expected:
            raise SchemaError("Collaboration schema columns or primary key conflict.")
    rows = connection.execute(
        "PRAGMA index_info(cayu_collaboration_event_participant_idx)"
    ).fetchall()
    if tuple(row[2] for row in rows) != ("scope", "participant_id", "sequence"):
        raise SchemaError("Collaboration event lookup index is unavailable.")
    if lifecycle or requests:
        indexes_to_check = {
            **(_LIFECYCLE_INDEXES if lifecycle else {}),
            **(_REQUEST_INDEXES if requests else {}),
        }
        for name, (table, columns, unique) in indexes_to_check.items():
            indexes = {row[1]: row for row in connection.execute(f"PRAGMA index_list({table})")}
            rows = connection.execute(f"PRAGMA index_info({name})").fetchall()
            if (
                tuple(row[2] for row in rows) != columns
                or name not in indexes
                or indexes[name][4] != 0
                or bool(indexes[name][2]) != unique
            ):
                raise SchemaError("Collaboration permit lookup index is unavailable.")


async def validate_postgres_collaboration_schema(
    cursor: Any, *, lifecycle: bool = False, requests: bool = False
) -> None:
    for table, columns, primary in _tables(lifecycle=lifecycle, requests=requests):
        await cursor.execute(
            """SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
            FROM pg_attribute a WHERE a.attrelid=to_regclass(%s)
            AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum""",
            (table,),
        )
        expected = [
            (name, "bigint" if name in _NUMERIC_COLUMNS else "text", True) for name in columns
        ]
        if await cursor.fetchall() != expected:
            raise SchemaError("Collaboration schema columns conflict.")
        await cursor.execute(
            """SELECT array_agg(a.attname ORDER BY k.ordinality)
            FROM pg_index i
            CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality)
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
            WHERE i.indrelid=to_regclass(%s) AND i.indisprimary AND i.indisvalid
            GROUP BY i.indexrelid""",
            (table,),
        )
        if await cursor.fetchall() != [(list(primary),)]:
            raise SchemaError("Collaboration schema primary key conflicts.")
    await cursor.execute("""SELECT array_agg(a.attname ORDER BY k.ordinality)
        FROM pg_index i
        CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality)
        JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
        WHERE i.indexrelid=to_regclass('cayu_collaboration_event_participant_idx')
        AND i.indrelid=to_regclass('cayu_collaboration_event_participants')
        AND i.indisvalid AND i.indpred IS NULL GROUP BY i.indexrelid""")
    if await cursor.fetchall() != [(["scope", "participant_id", "sequence"],)]:
        raise SchemaError("Collaboration event lookup index is unavailable.")
    if lifecycle or requests:
        indexes_to_check = {
            **(_LIFECYCLE_INDEXES if lifecycle else {}),
            **(_REQUEST_INDEXES if requests else {}),
        }
        for name, (table, columns, unique) in indexes_to_check.items():
            await cursor.execute(
                """SELECT array_agg(a.attname ORDER BY k.ordinality)
                FROM pg_index i
                CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality)
                JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
                WHERE i.indexrelid=to_regclass(%s)
                AND i.indrelid=to_regclass(%s)
                AND i.indisvalid AND i.indpred IS NULL AND i.indisunique=%s GROUP BY i.indexrelid""",
                (name, table, unique),
            )
            if await cursor.fetchall() != [(list(columns),)]:
                raise SchemaError("Collaboration permit lookup index is unavailable.")
