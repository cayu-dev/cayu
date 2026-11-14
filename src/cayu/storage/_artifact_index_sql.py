"""SQL for ``cayu_artifact_index`` reads and writes, in SQLite's and Postgres's spelling.

Every write carries the time its writer observed the artifact (``indexed_at``):
the commit hook uses the moment its ``put_bytes`` or ``delete`` finished, a
reindex the moment its walk began. A write applies only when it is at least as
new as the row, so a slow hook or walk never undoes a later deletion or commit,
and a walk never outranks a deletion that finished while it ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from cayu.artifacts._index import (
    ArtifactIndexQuery,
    ArtifactIndexValues,
    format_index_time,
)
from cayu.storage._artifact_index_schema import postgres_session_title, sqlite_session_title

Dialect = Literal["sqlite", "postgres"]

_COLUMNS = (
    "artifact_store_id",
    "artifact_id",
    "scope",
    "session_id",
    "agent_name",
    "environment_name",
    "created_at",
    "filename",
    "filename_key",
    "content_type",
    "size_bytes",
    "search_text",
    "metadata_json",
    "indexed_at",
)
_UPDATED = (*_COLUMNS[2:], "title")


@dataclass(frozen=True)
class _Spelling:
    dialect: Dialect

    @property
    def mark(self) -> str:
        return "?" if self.dialect == "sqlite" else "%s"

    def time(self, value: datetime) -> Any:
        return format_index_time(value) if self.dialect == "sqlite" else value

    def any_of(self, column: str, values: Sequence[str]) -> tuple[str, list[Any]]:
        if self.dialect == "sqlite":
            return f"{column} IN ({', '.join('?' for _ in values)})", list(values)
        return f"{column} = ANY(%s::text[])", [list(values)]

    def title_of(self, session_mark: str) -> str:
        title = sqlite_session_title if self.dialect == "sqlite" else postgres_session_title
        return (
            f"(SELECT {title('owner')} FROM cayu_sessions AS owner WHERE owner.id = {session_mark})"
        )


def _spelling(dialect: Dialect) -> _Spelling:
    return _Spelling(dialect)


def upsert_live_sql(dialect: Dialect) -> str:
    s = _spelling(dialect)
    marks = ", ".join(s.mark for _ in _COLUMNS)
    updates = ", ".join(f"{column} = excluded.{column}" for column in _UPDATED)
    return f"""
        INSERT INTO cayu_artifact_index ({", ".join(_COLUMNS)}, title, deleted_at)
        VALUES ({marks}, {s.title_of(s.mark)}, NULL)
        ON CONFLICT (artifact_store_id, artifact_id) DO UPDATE SET
            {updates}, deleted_at = NULL
        WHERE excluded.indexed_at >= cayu_artifact_index.indexed_at
    """


def upsert_live_params(
    dialect: Dialect, values: ArtifactIndexValues, observed_at: datetime
) -> tuple[Any, ...]:
    s = _spelling(dialect)
    return (
        values.store_id,
        values.artifact_id,
        values.scope,
        values.session_id,
        values.agent_name,
        values.environment_name,
        s.time(values.created_at),
        values.filename,
        values.filename_key,
        values.content_type,
        values.size_bytes,
        values.search_text,
        values.metadata_json,
        s.time(observed_at),
        values.session_id,
    )


def upsert_tombstone_sql(dialect: Dialect) -> str:
    mark = _spelling(dialect).mark
    return f"""
        INSERT INTO cayu_artifact_index (artifact_store_id, artifact_id, indexed_at, deleted_at)
        VALUES ({mark}, {mark}, {mark}, {mark})
        ON CONFLICT (artifact_store_id, artifact_id) DO UPDATE SET
            indexed_at = excluded.indexed_at, deleted_at = excluded.deleted_at
        WHERE excluded.indexed_at >= cayu_artifact_index.indexed_at
    """


def upsert_tombstone_params(
    dialect: Dialect, store_id: str, artifact_id: str, deleted_at: datetime
) -> tuple[Any, ...]:
    s = _spelling(dialect)
    return (store_id, artifact_id, s.time(deleted_at), s.time(deleted_at))


def mark_store_sql(dialect: Dialect) -> str:
    mark = _spelling(dialect).mark
    return f"""
        INSERT INTO cayu_artifact_index_stores (artifact_store_id, backfilled_at)
        VALUES ({mark}, {mark})
        ON CONFLICT (artifact_store_id) DO UPDATE SET backfilled_at = excluded.backfilled_at
    """


def mark_store_params(
    dialect: Dialect, store_id: str, backfilled_at: datetime | None
) -> tuple[Any, ...]:
    s = _spelling(dialect)
    return (store_id, None if backfilled_at is None else s.time(backfilled_at))


def claim_key(store_id: str) -> str:
    """The row holding the store's newest fill claim.

    A store ID never ends with whitespace, so this key is never a store's own.
    """

    return store_id + "\x1f"


def delete_store_row_sql(dialect: Dialect) -> str:
    mark = _spelling(dialect).mark
    return f"DELETE FROM cayu_artifact_index_stores WHERE artifact_store_id = {mark}"


def unfill_store_steps(store_id: str) -> tuple[tuple[Any, ...], ...]:
    """Forget the store's fill claim, then its filled marker, as separate statements.

    Reads then walk the store, and no fill that began earlier can complete. The order
    matters on PostgreSQL (READ COMMITTED): a fill completing concurrently takes back
    its claim before it writes the marker, so the claim delete waits for that fill to
    commit, and the marker delete, with a fresh snapshot, then sees its marker.
    """

    return ((claim_key(store_id),), (store_id,))


def claim_fill_sql(dialect: Dialect) -> str:
    """Record a fill's claim (the time it began), replacing any older fill's claim."""

    mark = _spelling(dialect).mark
    return f"""
        INSERT INTO cayu_artifact_index_stores (artifact_store_id, backfilled_at)
        VALUES ({mark}, {mark})
        ON CONFLICT (artifact_store_id) DO UPDATE SET backfilled_at = excluded.backfilled_at
    """


def claim_fill_params(dialect: Dialect, store_id: str, claimed_at: datetime) -> tuple[Any, ...]:
    return (claim_key(store_id), _spelling(dialect).time(claimed_at))


def release_claim_sql(dialect: Dialect) -> str:
    """Take back a fill's claim if it is still the store's newest and nothing unfilled
    the store since; a fill completes only when this removes a row."""

    mark = _spelling(dialect).mark
    return f"""
        DELETE FROM cayu_artifact_index_stores
        WHERE artifact_store_id = {mark} AND backfilled_at = {mark}
    """


def read_stores_sql(dialect: Dialect, store_ids: Sequence[str]) -> tuple[str, list[Any]]:
    clause, params = _spelling(dialect).any_of("artifact_store_id", store_ids)
    return (
        f"SELECT artifact_store_id, backfilled_at FROM cayu_artifact_index_stores WHERE {clause}",
        params,
    )


def scan_sql(dialect: Dialect, store_id: str, after: str | None, limit: int):
    mark = _spelling(dialect).mark
    params: list[Any] = [store_id]
    after_clause = ""
    if after is not None:
        after_clause = f"AND artifact_id > {mark}"
        params.append(after)
    params.append(limit)
    return (
        f"""SELECT artifact_id, metadata_json FROM cayu_artifact_index
        WHERE artifact_store_id = {mark} AND deleted_at IS NULL {after_clause}
        ORDER BY artifact_id LIMIT {mark}""",
        params,
    )


def count_sessions_sql(
    dialect: Dialect, store_ids: Sequence[str], session_ids: Sequence[str]
) -> tuple[str, list[Any]]:
    s = _spelling(dialect)
    sessions, session_params = s.any_of("session_id", session_ids)
    stores, store_params = s.any_of("artifact_store_id", store_ids)
    return (
        f"""SELECT session_id, COUNT(*) FROM cayu_artifact_index
        WHERE deleted_at IS NULL AND scope = 'session' AND {sessions} AND {stores}
        GROUP BY session_id""",
        [*session_params, *store_params],
    )


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filters(dialect: Dialect, query: ArtifactIndexQuery) -> tuple[list[str], list[Any]]:
    s = _spelling(dialect)
    stores, params = s.any_of("artifact_store_id", query.store_ids)
    clauses = ["deleted_at IS NULL", stores]
    for column, value in (
        ("scope", None if query.scope is None else query.scope.value),
        ("session_id", query.session_id),
        ("agent_name", query.agent_name),
        ("environment_name", query.environment_name),
    ):
        if value is not None:
            clauses.append(f"{column} = {s.mark}")
            params.append(value)
    needle = query.q
    if needle is not None:
        if dialect == "sqlite" and len(needle) >= 3:
            # Trigram full-text search: a quoted phrase matches a literal substring.
            clauses.append(
                "row_id IN (SELECT rowid FROM cayu_artifact_index_fts "
                "WHERE cayu_artifact_index_fts MATCH ?)"
            )
            params.append('"' + needle.replace('"', '""') + '"')
        elif dialect == "sqlite":
            clauses.append("(instr(search_text, ?) > 0 OR instr(lower(title), ?) > 0)")
            params.extend((needle, needle))
        else:
            pattern = "%" + _escape_like(needle) + "%"
            clauses.append("(search_text LIKE %s ESCAPE '\\' OR lower(title) LIKE %s ESCAPE '\\')")
            params.extend((pattern, pattern))
    return clauses, params


def page_sql(dialect: Dialect, query: ArtifactIndexQuery) -> tuple[str, list[Any]]:
    """The page's rows, one more than its limit, to tell whether another page follows."""

    s = _spelling(dialect)
    clauses, params = _filters(dialect, query)
    column = query.sort_column
    direction = "DESC" if query.descending else "ASC"
    if query.after is not None:
        # The cursor names its row; the row's sort value is read here. Rows are
        # never removed, so the row is still there after it is deleted.
        operator = "<" if query.descending else ">"
        anchor = (
            f"(SELECT anchor.{column} FROM cayu_artifact_index AS anchor "
            f"WHERE anchor.artifact_store_id = {s.mark} AND anchor.artifact_id = {s.mark})"
        )
        clauses.append(
            f"({column}, artifact_store_id, artifact_id) {operator} ({anchor}, {s.mark}, {s.mark})"
        )
        after = query.after
        params.extend((after.store_id, after.artifact_id, after.store_id, after.artifact_id))
    params.extend((query.limit + 1, query.offset))
    return (
        f"""SELECT artifact_store_id, metadata_json, title FROM cayu_artifact_index
        WHERE {" AND ".join(clauses)}
        ORDER BY {column} {direction}, artifact_store_id {direction}, artifact_id {direction}
        LIMIT {s.mark} OFFSET {s.mark}""",
        params,
    )


def total_sql(dialect: Dialect, query: ArtifactIndexQuery) -> tuple[str, list[Any]]:
    clauses, params = _filters(dialect, query)
    return f"SELECT COUNT(*) FROM cayu_artifact_index WHERE {' AND '.join(clauses)}", params
