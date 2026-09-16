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
}


def _ddl(postgres: bool) -> tuple[str, ...]:
    statements = []
    for family, columns in KEYS.items():
        fields = ["scope TEXT NOT NULL"]
        for column in columns:
            sql_type = "BIGINT" if column in ("generation", "revision", "sequence") else "TEXT"
            fields.append(f"{column} {sql_type} NOT NULL")
        check = "jsonb_typeof(document::jsonb) = 'object'" if postgres else "json_valid(document)"
        fields += [
            f"document TEXT NOT NULL CHECK ({check})",
            f"PRIMARY KEY ({', '.join(('scope', *columns))})",
        ]
        statements.append(
            f"CREATE TABLE IF NOT EXISTS cayu_collaboration_{family} ({', '.join(fields)})"
        )
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


def _tables():
    for family, columns in KEYS.items():
        primary = ("scope", *columns)
        yield f"cayu_collaboration_{family}", (*primary, "document"), primary
    primary = ("scope", "sequence", "participant_id")
    yield "cayu_collaboration_event_participants", primary, primary


def validate_sqlite_collaboration_schema(connection: Any) -> None:
    for table, columns, primary in _tables():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        expected = [
            (
                name,
                "BIGINT" if name in ("generation", "revision", "sequence") else "TEXT",
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


async def validate_postgres_collaboration_schema(cursor: Any) -> None:
    for table, columns, primary in _tables():
        await cursor.execute(
            """SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
            FROM pg_attribute a WHERE a.attrelid=to_regclass(%s)
            AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum""",
            (table,),
        )
        expected = [
            (name, "bigint" if name in ("generation", "revision", "sequence") else "text", True)
            for name in columns
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
