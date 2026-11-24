"""PostgreSQL store for tool-call preauthorizations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeVar

from psycopg.rows import tuple_row

from cayu.approvals import tool_call_preauthorizations as rules
from cayu.approvals.authorization_grants import (
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantNotFound,
)
from cayu.approvals.tool_call_preauthorizations import (
    ToolCallPreauthorization,
    ToolCallPreauthorizationStore,
    ToolCallPreauthorizationUse,
)
from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage._tool_call_preauthorization_schema import (
    TOOL_CALL_PREAUTHORIZATION_REVISION,
    TOOL_CALL_PREAUTHORIZATIONS_TABLE,
)
from cayu.storage.tool_call_preauthorizations_sqlite import preauthorization_row

_T = TypeVar("_T")
_COLUMNS = "preauthorization_id, task_id, tool_name, created_at_ms, record_json"
_ORDER = 'ORDER BY created_at_ms, preauthorization_id COLLATE "C"'


def _task_lock(task_id: str) -> str:
    return f"tool-call-preauthorization-task:{task_id}"


class PostgresToolCallPreauthorizationStore(_PostgresStoreBase, ToolCallPreauthorizationStore):
    """Durable preauthorizations shared by service processes on PostgreSQL.

    ``issue``, ``use`` and ``revoke`` each hold a transaction advisory lock on
    the record's task, so two workers can never use one record and a revoke
    never races a use. The
    table joins Cayu's shared revision history; the default ``schema_mode`` is
    ``VALIDATE`` and ``cayu storage migrate`` creates it.
    """

    _min_required_revision = TOOL_CALL_PREAUTHORIZATION_REVISION

    async def _locked(self, key: str, operation: Callable[[Any], Awaitable[_T]]) -> _T:
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor(row_factory=tuple_row) as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,)
                    )
                    result = await operation(cur)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return result

    @staticmethod
    async def _load(cur: Any, preauthorization_id: str) -> ToolCallPreauthorization | None:
        await cur.execute(
            f"SELECT record_json FROM {TOOL_CALL_PREAUTHORIZATIONS_TABLE} "
            "WHERE preauthorization_id = %s",
            (preauthorization_id,),
        )
        row = await cur.fetchone()
        return None if row is None else rules.preauthorization_from_record_json(row[0])

    @staticmethod
    async def _for_task(
        cur: Any, task_id: str, tool_name: str | None = None
    ) -> tuple[ToolCallPreauthorization, ...]:
        sql = f"SELECT record_json FROM {TOOL_CALL_PREAUTHORIZATIONS_TABLE} WHERE task_id = %s"
        values: tuple[str, ...] = (task_id,)
        if tool_name is not None:
            sql += " AND tool_name = %s"
            values = (task_id, tool_name)
        await cur.execute(f"{sql} {_ORDER}", values)
        rows = await cur.fetchall()
        return tuple(rules.preauthorization_from_record_json(row[0]) for row in rows)

    @staticmethod
    async def _write(cur: Any, record: ToolCallPreauthorization) -> None:
        await cur.execute(
            f"INSERT INTO {TOOL_CALL_PREAUTHORIZATIONS_TABLE} ({_COLUMNS}) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (preauthorization_id) "
            "DO UPDATE SET record_json = EXCLUDED.record_json",
            preauthorization_row(record),
        )

    async def issue(self, preauthorization: ToolCallPreauthorization) -> ToolCallPreauthorization:
        identifier = getattr(preauthorization, "preauthorization_id", "")

        async def operation(cur: Any) -> ToolCallPreauthorization:
            # Two issues of one id for different tasks hold different task locks; the id
            # lock makes the second see the first and conflict instead of overwriting it.
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"tool-call-preauthorization-id:{identifier}",),
            )
            record, changed = rules.issued_transition(
                await self._load(cur, identifier), preauthorization
            )
            if changed:
                await self._write(cur, record)
            return record

        task_id = getattr(preauthorization, "task_id", "")
        return await self._locked(_task_lock(task_id), operation)

    async def get(self, preauthorization_id: str) -> ToolCallPreauthorization:
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            record = await self._load(cur, preauthorization_id)
            await conn.commit()
        if record is None:
            raise ToolAuthorizationGrantNotFound(preauthorization_id)
        return record

    async def list_for_task(self, task_id: str) -> tuple[ToolCallPreauthorization, ...]:
        async with self._ready_connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            records = await self._for_task(cur, task_id)
            await conn.commit()
        return records

    async def use(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments_digest: str,
        agent_name: str,
        use: ToolCallPreauthorizationUse,
    ) -> ToolCallPreauthorization | None:
        async def operation(cur: Any) -> ToolCallPreauthorization | None:
            record, changed = rules.used_transition(
                await self._for_task(cur, task_id, tool_name),
                task_id=task_id,
                tool_name=tool_name,
                arguments_digest=arguments_digest,
                agent_name=agent_name,
                use=use,
            )
            if record is not None and changed:
                await self._write(cur, record)
            return record

        return await self._locked(_task_lock(task_id), operation)

    async def revoke(
        self,
        preauthorization_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        now: datetime,
    ) -> ToolCallPreauthorization:
        async def operation(cur: Any) -> ToolCallPreauthorization:
            record, changed = rules.revoked_transition(
                await self._load(cur, preauthorization_id),
                preauthorization_id=preauthorization_id,
                revoked_by=revoked_by,
                now=now,
            )
            if changed:
                await self._write(cur, record)
            return record

        # The task id never changes, so read it first and then lock that task.
        found = await self.get(preauthorization_id)
        return await self._locked(_task_lock(found.task_id), operation)


__all__ = ["PostgresToolCallPreauthorizationStore"]
