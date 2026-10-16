"""PostgreSQL tool authorization grant store."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from psycopg.rows import tuple_row

from cayu.approvals import authorization_grants as rules
from cayu.approvals.authorization_grants import (
    ToolAuthorizationGrant,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantStore,
    ToolAuthorizationGrantUse,
)
from cayu.storage._authorization_grant_schema import (
    TOOL_AUTHORIZATION_GRANT_REVISION,
    TOOL_AUTHORIZATION_GRANT_USES_TABLE,
    TOOL_AUTHORIZATION_GRANTS_TABLE,
)
from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage.authorization_grants_sqlite import (
    grant_list_filters,
    grant_row,
    use_cursor_clause,
)


class PostgresToolAuthorizationGrantStore(_PostgresStoreBase, ToolAuthorizationGrantStore):
    """Durable grants shared by service processes on PostgreSQL.

    Each mutation locks its grant row (or the absent identity, through a
    transaction advisory lock) and applies the same transition rules as the
    SQLite and in-memory stores. The tables join Cayu's shared revision history;
    the default ``schema_mode`` is ``VALIDATE`` and ``cayu storage migrate``
    creates or upgrades them.
    """

    _min_required_revision = TOOL_AUTHORIZATION_GRANT_REVISION

    async def _transition(
        self,
        grant_id: str,
        transition: Callable[[ToolAuthorizationGrant | None], tuple[ToolAuthorizationGrant, bool]],
    ) -> ToolAuthorizationGrant:
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"tool-authorization-grant:{grant_id}",),
                    )
                    await cur.execute(
                        f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANTS_TABLE} "
                        "WHERE grant_id = %s FOR UPDATE",
                        (grant_id,),
                    )
                    row = await cur.fetchone()
                    existing = None if row is None else rules.grant_from_record_json(row[0])
                    record, changed = transition(existing)
                    if changed:
                        await cur.execute(
                            f"INSERT INTO {TOOL_AUTHORIZATION_GRANTS_TABLE} "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT (grant_id) DO UPDATE SET status = EXCLUDED.status, "
                            "record_json = EXCLUDED.record_json",
                            grant_row(record),
                        )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return record

    async def stage(self, grant: ToolAuthorizationGrant) -> ToolAuthorizationGrant:
        grant = ToolAuthorizationGrant.model_validate(grant.model_dump(mode="python"))
        return await self._transition(
            grant.grant_id, lambda existing: rules.staged_grant_transition(existing, grant)
        )

    async def get(self, grant_id: str) -> ToolAuthorizationGrant | None:
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(
                f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANTS_TABLE} WHERE grant_id = %s",
                (grant_id,),
            )
            row = await cur.fetchone()
            await conn.commit()
        return None if row is None else rules.grant_from_record_json(row[0])

    async def list(self, query: ToolAuthorizationGrantQuery) -> tuple[ToolAuthorizationGrant, ...]:
        clauses, values = grant_list_filters(query, placeholder="%s", collate=' COLLATE "C"')
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = (
            f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANTS_TABLE} {where}"
            'ORDER BY created_at_ms, grant_id COLLATE "C" LIMIT %s'
        )
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(sql, (*values, query.limit))
            rows = await cur.fetchall()
            await conn.commit()
        return tuple(rules.grant_from_record_json(row[0]) for row in rows)

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
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(
                        f"INSERT INTO {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        (
                            *use.use_key,
                            rules.grant_timestamp_ms(use.recorded_at),
                            rules.use_record_json(use),
                        ),
                    )
                    await cur.execute(
                        f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                        "WHERE grant_id = %s AND session_id = %s AND session_instance_id = %s "
                        "AND tool_round_id = %s AND tool_call_id = %s AND arguments_digest = %s",
                        use.use_key,
                    )
                    row = await cur.fetchone()
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return rules.use_from_record_json(row[0])

    async def list_uses(
        self,
        grant_id: str,
        *,
        limit: int = 100,
        after: rules.ToolAuthorizationGrantUseCursor | None = None,
    ) -> tuple[ToolAuthorizationGrantUse, ...]:
        limit = rules.require_grant_list_limit(limit)
        position, values = use_cursor_clause(after, placeholder="%s", collate=' COLLATE "C"')
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            await cur.execute(
                f"SELECT record_json FROM {TOOL_AUTHORIZATION_GRANT_USES_TABLE} "
                f"WHERE grant_id = %s{position} ORDER BY recorded_at_ms, "
                'session_id COLLATE "C", session_instance_id COLLATE "C", '
                'tool_round_id COLLATE "C", tool_call_id COLLATE "C", '
                'arguments_digest COLLATE "C" LIMIT %s',
                (grant_id, *values, limit),
            )
            rows = await cur.fetchall()
            await conn.commit()
        return tuple(rules.use_from_record_json(row[0]) for row in rows)


__all__ = ["PostgresToolAuthorizationGrantStore"]
