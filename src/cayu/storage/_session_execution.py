"""Small epoch-fenced liveness rows, independent of checkpoint/event publication."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

from cayu._validation import require_clean_nonblank
from cayu.sessions.execution import _ExecutionOwner, execution_state

SQLITE_EXECUTION_DDL = """
CREATE TABLE IF NOT EXISTS cayu_session_execution_owners (
    session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
    owner_json TEXT NOT NULL CHECK (json_valid(owner_json) AND length(owner_json) <= 4096)
);
"""
POSTGRES_EXECUTION_DDL = (
    """
CREATE TABLE IF NOT EXISTS cayu_session_execution_owners (
    session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
    owner_json JSONB NOT NULL CHECK (octet_length(owner_json::text) <= 4096)
)
""",
)


def _identity(row):
    return None if row is None else SimpleNamespace(id=row[0], instance_id=row[1], run_epoch=row[2])


def _live(session, owner, now):
    return (
        owner is not None
        and session is not None
        and not owner.released
        and owner.session_instance_id == session.instance_id
        and owner.run_epoch == session.run_epoch
        and owner.lease_expires_at > now
    )


def _claim(
    session,
    old,
    *,
    now,
    token,
    owner_id,
    owner_kind,
    owner_label,
    lease_seconds,
    operation_id,
    operation_run_epoch,
):
    from cayu.sessions.base import _current_session_run_epoch

    if session is None or session.run_epoch != _current_session_run_epoch(session.id):
        return None
    if (
        old is not None
        and old.session_instance_id == session.instance_id
        and old.run_epoch == session.run_epoch
    ):
        if old.owner_id != owner_id:
            return None
        if not old.released and old.lease_expires_at > now:
            return old if old.token == token else None
        # The same process may recover observation after expiry. The transaction
        # still requires its current run epoch, so a successor fence wins.

    return _ExecutionOwner(
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        token=token,
        owner_id=owner_id,
        owner_kind=owner_kind,
        owner_label=owner_label,
        operation_id=None
        if operation_id is None or operation_run_epoch != session.run_epoch
        else "sha256:" + sha256(operation_id.encode()).hexdigest(),
        claimed_at=now,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        last_progress_at=now,
    )


def _renew(
    session,
    old,
    expected,
    *,
    now,
    lease_seconds,
    progress_kind,
    operation_id=None,
    operation_run_epoch=None,
):
    if (
        session is None
        or old is None
        or old.token != expected.token
        or old.released
        or session.instance_id != expected.session_instance_id
        or session.run_epoch != expected.run_epoch
        or old.lease_expires_at <= now
    ):
        return None
    fields = {"heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds)}
    if (
        old.operation_id is None
        and operation_id is not None
        and operation_run_epoch == expected.run_epoch
    ):
        fields["operation_id"] = "sha256:" + sha256(operation_id.encode()).hexdigest()
    if progress_kind is not None:
        fields.update(last_progress_at=now, last_progress_kind=progress_kind)
    return old.model_copy(update=fields)


class MemorySessionExecutionMixin:
    _lock: Any
    _sessions: Any
    _checkpoints: Any
    _execution_owners: dict[str, _ExecutionOwner]
    _ownership_clock: Any
    supports_session_execution = True

    def _has_live_execution_owner_unlocked(self, session, now):
        return _live(session, self._execution_owners.get(session.id), now)

    async def _claim_session_execution(self, session_id, **arguments):
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id) or {}
            operation = checkpoint.get("session_run_operation") or {}
            owner = _claim(
                self._sessions.get(session_id),
                self._execution_owners.get(session_id),
                now=self._ownership_clock(),
                operation_id=operation.get("operation_id"),
                operation_run_epoch=operation.get("run_epoch"),
                **arguments,
            )
            if owner is not None:
                self._execution_owners[session_id] = owner
            return owner

    async def _renew_session_execution(self, expected, **arguments):
        async with self._lock:
            operation = (self._checkpoints.get(expected.session_id) or {}).get(
                "session_run_operation"
            ) or {}
            owner = _renew(
                self._sessions.get(expected.session_id),
                self._execution_owners.get(expected.session_id),
                expected,
                now=self._ownership_clock(),
                operation_id=operation.get("operation_id"),
                operation_run_epoch=operation.get("run_epoch"),
                **arguments,
            )
            if owner is not None:
                self._execution_owners[expected.session_id] = owner
            return owner

    async def _release_session_execution(self, expected):
        async with self._lock:
            old = self._execution_owners.get(expected.session_id)
            if old is not None and old.token == expected.token:
                self._execution_owners[expected.session_id] = old.model_copy(
                    update={"released": True}
                )

    async def inspect_session_execution(self, session_id):
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            checkpoint = self._checkpoints.get(session_id) or {}
            return execution_state(
                session_id=session.id,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                status=session.status,
                waiting=any(
                    checkpoint.get(name) is not None
                    for name in ("pending_user_input", "pending_tool_approval")
                ),
                waiting_for_child=checkpoint.get("foreground_child_wait") is not None,
                owner=self._execution_owners.get(session_id),
                now=self._ownership_clock(),
            )


class SQLiteSessionExecutionMixin:
    _lock: Any
    _connection: Any
    _ownership_clock: Any
    _run_read: Any
    _run_write: Any
    supports_session_execution = True

    def _execution_identity(self, session_id, connection=None):
        connection = self._connection if connection is None else connection
        return _identity(
            connection.execute(
                "SELECT id, instance_id, run_epoch FROM cayu_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        )

    def _has_live_execution_owner_unlocked(self, session, now):
        row = self._connection.execute(
            "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = ?",
            (session.id,),
        ).fetchone()
        return _live(
            session, None if row is None else _ExecutionOwner.model_validate_json(row[0]), now
        )

    async def _claim_session_execution(self, session_id, **arguments):
        def statement(connection):
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                operation = connection.execute(
                    "SELECT json_extract(state_json, '$.session_run_operation.operation_id'), json_extract(state_json, '$.session_run_operation.run_epoch') FROM cayu_checkpoints WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                owner = _claim(
                    self._execution_identity(session_id, connection),
                    None if row is None else _ExecutionOwner.model_validate_json(row[0]),
                    now=self._ownership_clock(),
                    operation_id=None if operation is None else operation[0],
                    operation_run_epoch=None if operation is None else operation[1],
                    **arguments,
                )
                if owner is not None:
                    connection.execute(
                        "INSERT INTO cayu_session_execution_owners VALUES (?, ?) ON CONFLICT (session_id) DO UPDATE SET owner_json = excluded.owner_json",
                        (session_id, owner.model_dump_json()),
                    )
                connection.commit()
                return owner
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _renew_session_execution(self, expected, **arguments):
        def statement(connection):
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = ?",
                    (expected.session_id,),
                ).fetchone()
                operation = (
                    connection.execute(
                        "SELECT json_extract(state_json, '$.session_run_operation.operation_id'), json_extract(state_json, '$.session_run_operation.run_epoch') FROM cayu_checkpoints WHERE session_id = ?",
                        (expected.session_id,),
                    ).fetchone()
                    if expected.operation_id is None
                    else None
                )
                owner = _renew(
                    self._execution_identity(expected.session_id, connection),
                    None if row is None else _ExecutionOwner.model_validate_json(row[0]),
                    expected,
                    now=self._ownership_clock(),
                    operation_id=None if operation is None else operation[0],
                    operation_run_epoch=None if operation is None else operation[1],
                    **arguments,
                )
                if owner is not None:
                    connection.execute(
                        "UPDATE cayu_session_execution_owners SET owner_json = ? WHERE session_id = ?",
                        (owner.model_dump_json(), expected.session_id),
                    )
                connection.commit()
                return owner
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _release_session_execution(self, expected):
        def statement(connection):
            try:
                connection.execute(
                    "UPDATE cayu_session_execution_owners SET owner_json = json_set(owner_json, '$.released', json('true')) WHERE session_id = ? AND json_extract(owner_json, '$.token') = ?",
                    (expected.session_id, expected.token),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def inspect_session_execution(self, session_id):
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection):
            row = connection.execute(
                "SELECT s.id, s.instance_id, s.run_epoch, s.status, c.pending_action_flags, o.owner_json FROM cayu_sessions s LEFT JOIN cayu_checkpoints c ON c.session_id = s.id LEFT JOIN cayu_session_execution_owners o ON o.session_id = s.id WHERE s.id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            return execution_state(
                session_id=row[0],
                session_instance_id=row[1],
                run_epoch=row[2],
                status=row[3],
                waiting=bool((row[4] or 0) & 3),
                waiting_for_child=bool((row[4] or 0) & 8),
                owner=None if row[5] is None else _ExecutionOwner.model_validate_json(row[5]),
                now=self._ownership_clock(),
            )

        return await self._run_read(query)


class PostgresSessionExecutionMixin:
    _ensure_ready: Any
    _connection: Any
    supports_session_execution = True

    @staticmethod
    async def _session_store_now(cur: Any) -> datetime:
        raise NotImplementedError

    async def _execution_identity(self, cur, session_id):
        await cur.execute(
            "SELECT id, instance_id, run_epoch FROM cayu_sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        )
        return _identity(await cur.fetchone())

    async def _has_live_execution_owner(self, cur, session, now):
        await cur.execute(
            "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = %s",
            (session.id,),
        )
        row = await cur.fetchone()
        return _live(session, None if row is None else _ExecutionOwner.model_validate(row[0]), now)

    async def _claim_session_execution(self, session_id, **arguments):
        from psycopg.types.json import Jsonb

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            session = await self._execution_identity(cur, session_id)
            await cur.execute(
                "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            await cur.execute(
                "SELECT state->'session_run_operation'->>'operation_id', (state->'session_run_operation'->>'run_epoch')::BIGINT FROM cayu_checkpoints WHERE session_id = %s",
                (session_id,),
            )
            operation = await cur.fetchone()
            owner = _claim(
                session,
                None if row is None else _ExecutionOwner.model_validate(row[0]),
                now=await self._session_store_now(cur),
                operation_id=None if operation is None else operation[0],
                operation_run_epoch=None if operation is None else operation[1],
                **arguments,
            )
            if owner is not None:
                await cur.execute(
                    "INSERT INTO cayu_session_execution_owners VALUES (%s, %s) ON CONFLICT (session_id) DO UPDATE SET owner_json = excluded.owner_json",
                    (session_id, Jsonb(owner.model_dump(mode="json"))),
                )
            await conn.commit()
            return owner

    async def _renew_session_execution(self, expected, **arguments):
        from psycopg.types.json import Jsonb

        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            session = await self._execution_identity(cur, expected.session_id)
            await cur.execute(
                "SELECT owner_json FROM cayu_session_execution_owners WHERE session_id = %s",
                (expected.session_id,),
            )
            row = await cur.fetchone()
            operation = None
            if expected.operation_id is None:
                await cur.execute(
                    "SELECT state->'session_run_operation'->>'operation_id', (state->'session_run_operation'->>'run_epoch')::BIGINT FROM cayu_checkpoints WHERE session_id = %s",
                    (expected.session_id,),
                )
                operation = await cur.fetchone()
            owner = _renew(
                session,
                None if row is None else _ExecutionOwner.model_validate(row[0]),
                expected,
                now=await self._session_store_now(cur),
                operation_id=None if operation is None else operation[0],
                operation_run_epoch=None if operation is None else operation[1],
                **arguments,
            )
            if owner is not None:
                await cur.execute(
                    "UPDATE cayu_session_execution_owners SET owner_json = %s WHERE session_id = %s",
                    (Jsonb(owner.model_dump(mode="json")), expected.session_id),
                )
            await conn.commit()
            return owner

    async def _release_session_execution(self, expected):
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE cayu_session_execution_owners SET owner_json = jsonb_set(owner_json, '{released}', 'true') WHERE session_id = %s AND owner_json->>'token' = %s",
                (expected.session_id, expected.token),
            )
            await conn.commit()

    async def inspect_session_execution(self, session_id):
        session_id = require_clean_nonblank(session_id, "session_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT s.id, s.instance_id, s.run_epoch, s.status, c.pending_action_flags, o.owner_json FROM cayu_sessions s LEFT JOIN cayu_checkpoints c ON c.session_id = s.id LEFT JOIN cayu_session_execution_owners o ON o.session_id = s.id WHERE s.id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            now = await self._session_store_now(cur)
            return execution_state(
                session_id=row[0],
                session_instance_id=row[1],
                run_epoch=row[2],
                status=row[3],
                waiting=bool((row[4] or 0) & 3),
                waiting_for_child=bool((row[4] or 0) & 8),
                owner=None if row[5] is None else _ExecutionOwner.model_validate(row[5]),
                now=now,
            )
