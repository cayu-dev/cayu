"""Structural qualification of the native selection/exclusion decision owner."""

from typing import Any

from cayu.storage.migrations import SchemaError

_TABLE = "cayu_context_selection_exclusions"
_INDEX = "idx_context_selection_exclusions_owner"
_LOOKUP = ("owner_scope", "owner_id", "owner_incarnation")
_COLUMNS = (
    "selection_key",
    *_LOOKUP,
    "source_session_id",
    "source_session_instance_id",
    "request_commitment",
    "decision_json",
)


def _conflict() -> SchemaError:
    return SchemaError(
        "Context selection fence schema is missing or conflicts with its required "
        "table/index contract. Apply revision 108 or restore a known-good schema."
    )


def validate_sqlite_context_selection_schema(connection: Any) -> None:
    rows = connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
    actual = {row[1]: (*row[2:4], row[5]) for row in rows}
    if any(
        actual.get(name) != ("TEXT", int(name != "selection_key"), int(name == "selection_key"))
        for name in _COLUMNS
    ):
        raise _conflict()
    primary = lookup = False
    for row in connection.execute(f"PRAGMA index_list({_TABLE})").fetchall():
        name = str(row[1]).replace("'", "''")
        columns = tuple(r[2] for r in connection.execute(f"PRAGMA index_info('{name}')"))
        primary |= bool(row[2] and row[3] == "pk" and not row[4] and columns == ("selection_key",))
        lookup |= bool(row[1] == _INDEX and not row[2] and not row[4] and columns == _LOOKUP)
    if not primary or not lookup:
        raise _conflict()


async def validate_postgres_context_selection_schema(cursor: Any) -> None:
    await cursor.execute(
        """SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
        FROM pg_attribute a WHERE a.attrelid=to_regclass(%s)
        AND a.attnum > 0 AND NOT a.attisdropped""",
        (_TABLE,),
    )
    actual = {name: (kind, required) for name, kind, required in await cursor.fetchall()}
    if any(actual.get(name) != ("text", True) for name in _COLUMNS):
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
    if not any(primary and tuple(columns) == ("selection_key",) for _, _, primary, columns in rows):
        raise _conflict()
    if not any(
        name == _INDEX and not unique and tuple(columns) == _LOOKUP
        for name, unique, _, columns in rows
    ):
        raise _conflict()
