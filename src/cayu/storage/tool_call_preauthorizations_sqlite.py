"""SQLite store for tool-call preauthorizations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from cayu.approvals import tool_call_preauthorizations as rules
from cayu.approvals.authorization_grants import (
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantNotFound,
    grant_timestamp_ms,
)
from cayu.approvals.tool_call_preauthorizations import (
    ToolCallPreauthorization,
    ToolCallPreauthorizationStore,
    ToolCallPreauthorizationUse,
)
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage._tool_call_preauthorization_schema import (
    TOOL_CALL_PREAUTHORIZATION_REVISION,
    TOOL_CALL_PREAUTHORIZATIONS_TABLE,
)
from cayu.storage.targets import require_sqlite_store_allowed

_T = TypeVar("_T")
_COLUMNS = "preauthorization_id, task_id, tool_name, created_at_ms, record_json"


def preauthorization_row(record: ToolCallPreauthorization) -> tuple[Any, ...]:
    """Indexed projection plus the authoritative record, in column order."""

    return (
        record.preauthorization_id,
        record.task_id,
        record.tool_name,
        grant_timestamp_ms(record.created_at),
        rules.preauthorization_record_json(record),
    )


class SQLiteToolCallPreauthorizationStore(ToolCallPreauthorizationStore):
    """Durable preauthorizations on SQLite, in Cayu's shared schema history.

    ``issue``, ``use`` and ``revoke`` each read and write inside one
    ``BEGIN IMMEDIATE`` transaction with the shared transition rules, so two
    calls can never use one record. ``CREATE`` initializes an empty database
    and otherwise validates; an older database needs ``cayu storage migrate``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        require_sqlite_store_allowed("SQLiteToolCallPreauthorizationStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteToolCallPreauthorizationStore path must be a nonblank path.")
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
                app_min_supported=TOOL_CALL_PREAUTHORIZATION_REVISION,
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
            raise RuntimeError("SQLiteToolCallPreauthorizationStore is closed.")
        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    @staticmethod
    def _load(connection: sqlite3.Connection, preauthorization_id: str):
        row = connection.execute(
            f"SELECT record_json FROM {TOOL_CALL_PREAUTHORIZATIONS_TABLE} "
            "WHERE preauthorization_id = ?",
            (preauthorization_id,),
        ).fetchone()
        return None if row is None else rules.preauthorization_from_record_json(row[0])

    @staticmethod
    def _for_task(connection: sqlite3.Connection, task_id: str, tool_name: str | None = None):
        sql = f"SELECT record_json FROM {TOOL_CALL_PREAUTHORIZATIONS_TABLE} WHERE task_id = ?"
        values: tuple[str, ...] = (task_id,)
        if tool_name is not None:
            sql += " AND tool_name = ?"
            values = (task_id, tool_name)
        rows = connection.execute(
            sql + " ORDER BY created_at_ms, preauthorization_id", values
        ).fetchall()
        return tuple(rules.preauthorization_from_record_json(row[0]) for row in rows)

    @staticmethod
    def _write(connection: sqlite3.Connection, record: ToolCallPreauthorization) -> None:
        connection.execute(
            f"INSERT INTO {TOOL_CALL_PREAUTHORIZATIONS_TABLE} ({_COLUMNS}) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (preauthorization_id) DO UPDATE SET record_json = excluded.record_json",
            preauthorization_row(record),
        )

    async def issue(self, preauthorization: ToolCallPreauthorization) -> ToolCallPreauthorization:
        def operation(connection: sqlite3.Connection) -> ToolCallPreauthorization:
            with sqlite_connection._transaction(connection):
                existing = self._load(
                    connection, getattr(preauthorization, "preauthorization_id", "")
                )
                record, changed = rules.issued_transition(existing, preauthorization)
                if changed:
                    self._write(connection, record)
                return record

        return await self._run(operation)

    async def get(self, preauthorization_id: str) -> ToolCallPreauthorization:
        record = await self._run(lambda connection: self._load(connection, preauthorization_id))
        if record is None:
            raise ToolAuthorizationGrantNotFound(preauthorization_id)
        return record

    async def list_for_task(self, task_id: str) -> tuple[ToolCallPreauthorization, ...]:
        return await self._run(lambda connection: self._for_task(connection, task_id))

    async def use(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments_digest: str,
        agent_name: str,
        use: ToolCallPreauthorizationUse,
    ) -> ToolCallPreauthorization | None:
        def operation(connection: sqlite3.Connection) -> ToolCallPreauthorization | None:
            with sqlite_connection._transaction(connection):
                record, changed = rules.used_transition(
                    self._for_task(connection, task_id, tool_name),
                    task_id=task_id,
                    tool_name=tool_name,
                    arguments_digest=arguments_digest,
                    agent_name=agent_name,
                    use=use,
                )
                if record is not None and changed:
                    self._write(connection, record)
                return record

        return await self._run(operation)

    async def revoke(
        self,
        preauthorization_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        now: datetime,
    ) -> ToolCallPreauthorization:
        def operation(connection: sqlite3.Connection) -> ToolCallPreauthorization:
            with sqlite_connection._transaction(connection):
                record, changed = rules.revoked_transition(
                    self._load(connection, preauthorization_id),
                    preauthorization_id=preauthorization_id,
                    revoked_by=revoked_by,
                    now=now,
                )
                if changed:
                    self._write(connection, record)
                return record

        return await self._run(operation)


__all__ = ["SQLiteToolCallPreauthorizationStore", "preauthorization_row"]
