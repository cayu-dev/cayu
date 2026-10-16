"""SQLite tool authorization grant store."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from cayu.approvals import authorization_grants as rules
from cayu.approvals.authorization_grants import (
    ToolAuthorizationGrant,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantStore,
    ToolAuthorizationGrantUse,
)
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._authorization_grant_schema import (
    TOOL_AUTHORIZATION_GRANT_REVISION,
    TOOL_AUTHORIZATION_GRANT_USES_TABLE,
    TOOL_AUTHORIZATION_GRANTS_TABLE,
)
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage.targets import require_sqlite_store_allowed

_T = TypeVar("_T")


def grant_row(grant: ToolAuthorizationGrant) -> tuple[Any, ...]:
    """Indexed projection plus the authoritative record, in column order."""

    return (
        grant.grant_id,
        grant.policy_id,
        grant.tool_name,
        grant.scope.kind.value,
        grant.scope.id,
        grant.origin.session_id,
        grant.status.value,
        rules.grant_timestamp_ms(grant.created_at),
        rules.grant_record_json(grant),
    )


def grant_list_filters(
    query: ToolAuthorizationGrantQuery, *, placeholder: str = "?", collate: str = ""
) -> tuple[list[str], list[Any]]:
    """Exact-match WHERE clauses shared by the SQL grant stores."""

    clauses: list[str] = []
    values: list[Any] = []
    for column, value in (
        ("policy_id", query.policy_id),
        ("tool_name", query.tool_name),
        ("scope_kind", None if query.scope_kind is None else query.scope_kind.value),
        ("scope_id", query.scope_id),
        ("origin_session_id", query.origin_session_id),
    ):
        if value is not None:
            clauses.append(f"{column} = {placeholder}")
            values.append(value)
    if query.statuses is not None:
        if not query.statuses:
            clauses.append("1 = 0")
        else:
            marks = ", ".join(placeholder for _ in query.statuses)
            clauses.append(f"status IN ({marks})")
            values.extend(status.value for status in query.statuses)
    after = query.after()
    if after is not None:
        created_at_ms = rules.grant_timestamp_ms(after[0])
        clauses.append(
            f"(created_at_ms > {placeholder} OR "
            f"(created_at_ms = {placeholder} AND grant_id{collate} > {placeholder}))"
        )
        values.extend((created_at_ms, created_at_ms, after[1]))
    return clauses, values


def use_cursor_clause(
    after: rules.ToolAuthorizationGrantUseCursor | None, *, placeholder: str, collate: str = ""
) -> tuple[str, tuple[Any, ...]]:
    """The exclusive use-listing position shared by the SQL grant stores."""

    if after is None:
        return "", ()
    columns = ", ".join(
        [
            "recorded_at_ms",
            *(
                f"{column}{collate}"
                for column in (
                    "session_id",
                    "session_instance_id",
                    "tool_round_id",
                    "tool_call_id",
                    "arguments_digest",
                )
            ),
        ]
    )
    marks = ", ".join([placeholder] * 6)
    key = after.key()
    return f" AND ({columns}) > ({marks})", (rules.grant_timestamp_ms(key[0]), *key[1:])


class SQLiteToolAuthorizationGrantStore(ToolAuthorizationGrantStore):
    """Durable grants on SQLite, sharing Cayu's schema revision history.

    Every mutation reads and writes its row inside one ``BEGIN IMMEDIATE``
    transaction using the shared transition rules, so concurrent stage,
    activate, and revoke calls serialize. Like the other SQLite stores,
    ``CREATE`` initializes an empty database and otherwise validates; an older
    database requires ``cayu storage migrate``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        require_sqlite_store_allowed("SQLiteToolAuthorizationGrantStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteToolAuthorizationGrantStore path must be a nonblank path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._lock = TimedStoreLock()
        self._closed = False
        self._connection = sqlite_connection.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=TOOL_AUTHORIZATION_GRANT_REVISION,
            )
        except BaseException:
            self._connection.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, lambda connection: connection.close()
        )

    async def _run(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        if self._closed:
            raise RuntimeError("SQLiteToolAuthorizationGrantStore is closed.")
        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    @staticmethod
    def _load(connection: sqlite3.Connection, grant_id: str) -> ToolAuthorizationGrant | None:
        row = connection.execute(
            f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANTS_TABLE} WHERE grant_id = ?",
            (grant_id,),
        ).fetchone()
        return None if row is None else rules.grant_from_record_json(row[0])

    @staticmethod
    def _write(connection: sqlite3.Connection, grant: ToolAuthorizationGrant) -> None:
        connection.execute(
            f"INSERT INTO {TOOL_AUTHORIZATION_GRANTS_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (grant_id) DO UPDATE SET status = excluded.status, "
            "record_json = excluded.record_json",
            grant_row(grant),
        )

    async def _transition(
        self,
        grant_id: str,
        transition: Callable[[ToolAuthorizationGrant | None], tuple[ToolAuthorizationGrant, bool]],
    ) -> ToolAuthorizationGrant:
        def operation(connection: sqlite3.Connection) -> ToolAuthorizationGrant:
            with sqlite_connection._transaction(connection):
                record, changed = transition(self._load(connection, grant_id))
                if changed:
                    self._write(connection, record)
                return record

        return await self._run(operation)

    async def stage(self, grant: ToolAuthorizationGrant) -> ToolAuthorizationGrant:
        grant = ToolAuthorizationGrant.model_validate(grant.model_dump(mode="python"))
        return await self._transition(
            grant.grant_id, lambda existing: rules.staged_grant_transition(existing, grant)
        )

    async def get(self, grant_id: str) -> ToolAuthorizationGrant | None:
        return await self._run(lambda connection: self._load(connection, grant_id))

    async def list(self, query: ToolAuthorizationGrantQuery) -> tuple[ToolAuthorizationGrant, ...]:
        clauses, values = grant_list_filters(query)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = (
            f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANTS_TABLE} {where}"
            "ORDER BY created_at_ms, grant_id LIMIT ?"
        )

        def operation(connection: sqlite3.Connection) -> tuple[ToolAuthorizationGrant, ...]:
            rows = connection.execute(sql, (*values, query.limit)).fetchall()
            return tuple(rules.grant_from_record_json(row[0]) for row in rows)

        return await self._run(operation)

    async def activate(
        self,
        grant_id: str,
        *,
        resolved_by: ToolAuthorizationGrantActor | None,
        activated_at: datetime,
    ) -> ToolAuthorizationGrant:
        return await self._transition(
            grant_id,
            lambda existing: rules.activated_grant_transition(
                existing, grant_id=grant_id, resolved_by=resolved_by, activated_at=activated_at
            ),
        )

    async def abandon(
        self,
        grant_id: str,
        *,
        reason: str,
        abandoned_at: datetime,
        expected_request_digests: tuple[str, ...] | None = None,
    ) -> ToolAuthorizationGrant:
        return await self._transition(
            grant_id,
            lambda existing: rules.abandoned_grant_transition(
                existing,
                grant_id=grant_id,
                reason=reason,
                abandoned_at=abandoned_at,
                expected_request_digests=expected_request_digests,
            ),
        )

    async def revoke(
        self,
        grant_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        reason: str | None,
        revoked_at: datetime,
    ) -> ToolAuthorizationGrant:
        return await self._transition(
            grant_id,
            lambda existing: rules.revoked_grant_transition(
                existing,
                grant_id=grant_id,
                revoked_by=revoked_by,
                reason=reason,
                revoked_at=revoked_at,
            ),
        )

    async def record_use(self, use: ToolAuthorizationGrantUse) -> ToolAuthorizationGrantUse:
        use = ToolAuthorizationGrantUse.model_validate(use.model_dump(mode="python"))

        def operation(connection: sqlite3.Connection) -> ToolAuthorizationGrantUse:
            with sqlite_connection._transaction(connection):
                connection.execute(
                    f"INSERT INTO {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    (
                        *use.use_key,
                        rules.grant_timestamp_ms(use.recorded_at),
                        rules.use_record_json(use),
                    ),
                )
                row = connection.execute(
                    f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                    "WHERE grant_id = ? AND session_id = ? AND session_instance_id = ? "
                    "AND tool_round_id = ? AND tool_call_id = ? AND arguments_digest = ?",
                    use.use_key,
                ).fetchone()
                return rules.use_from_record_json(row[0])

        return await self._run(operation)

    async def list_uses(
        self,
        grant_id: str,
        *,
        limit: int = 100,
        after: rules.ToolAuthorizationGrantUseCursor | None = None,
    ) -> tuple[ToolAuthorizationGrantUse, ...]:
        limit = rules.require_grant_list_limit(limit)
        position, values = use_cursor_clause(after, placeholder="?")

        def operation(connection: sqlite3.Connection) -> tuple[ToolAuthorizationGrantUse, ...]:
            rows = connection.execute(
                f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                f"WHERE grant_id = ?{position} "
                "ORDER BY recorded_at_ms, session_id, session_instance_id, tool_round_id, "
                "tool_call_id, arguments_digest LIMIT ?",
                (grant_id, *values, limit),
            ).fetchall()
            return tuple(rules.use_from_record_json(row[0]) for row in rows)

        return await self._run(operation)


__all__ = ["SQLiteToolAuthorizationGrantStore"]
