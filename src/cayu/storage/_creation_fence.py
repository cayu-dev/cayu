"""Backend transactions for the SessionStore-owned future-target fence."""

from __future__ import annotations

from typing import Any

from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.sessions.creation_fence import (
    SessionCreationConflict,
    SessionCreationDecision,
    SessionCreationTarget,
    decide,
    discovery_bounds,
    discovery_page,
    owner_key,
    require_authority,
    snapshot_target,
)


class MemoryCreationFenceMixin:
    _lock: Any
    _session_creation_decisions: dict[str, SessionCreationDecision]

    async def list_pending_session_creations(self, owner, *, cursor=None, limit=32):
        discovery_bounds(cursor, limit)
        expected_owner = owner_key(owner)
        async with self._lock:
            decisions = [
                value
                for key, value in sorted(self._session_creation_decisions.items())
                if key > (cursor or "")
                and not value.settlement_acknowledged
                and owner_key(value.target.receiving_owner) == expected_owner
            ][: limit + 1]
            return discovery_page(decisions, limit)

    async def _prepare_session_creation_target(self, target, *, authority):
        return await memory_decision(self, target, authority=authority, mutate=True)

    async def _register_session_creation_target(self, target, *, authority):
        return await memory_decision(self, target, authority=authority, mutate=True, register=True)

    async def _exclude_session_creation_target(self, target, *, authority):
        return await memory_decision(self, target, authority=authority, mutate=True, exclude=True)

    async def _acknowledge_session_creation_settlement(self, target, *, authority):
        return await memory_decision(
            self, target, authority=authority, mutate=True, acknowledge=True
        )

    async def read_session_creation_decision(self, target):
        target = snapshot_target(target)
        return await exact_read(memory_decision(self, target))


class SQLiteCreationFenceMixin:
    _run_read: Any

    async def list_pending_session_creations(self, owner, *, cursor=None, limit=32):
        discovery_bounds(cursor, limit)
        expected_owner = owner_key(owner)

        def query(connection):
            rows = connection.execute(
                "SELECT decision_json FROM cayu_session_creation_decisions "
                "WHERE owner_key = ? AND recovery_pending = 1 AND operation_key > ? "
                "ORDER BY operation_key LIMIT ?",
                (expected_owner, cursor or "", limit + 1),
            ).fetchall()
            return discovery_page(
                [SessionCreationDecision.model_validate_json(row[0]) for row in rows], limit
            )

        return await self._run_read(query)

    async def _prepare_session_creation_target(self, target, *, authority):
        return await sqlite_decision(self, target, authority=authority, mutate=True)

    async def _register_session_creation_target(self, target, *, authority):
        return await sqlite_decision(self, target, authority=authority, mutate=True, register=True)

    async def _exclude_session_creation_target(self, target, *, authority):
        return await sqlite_decision(self, target, authority=authority, mutate=True, exclude=True)

    async def _acknowledge_session_creation_settlement(self, target, *, authority):
        return await sqlite_decision(
            self, target, authority=authority, mutate=True, acknowledge=True
        )

    async def read_session_creation_decision(self, target):
        target = snapshot_target(target)
        return await exact_read(sqlite_decision(self, target))


class PostgresCreationFenceMixin:
    _ensure_ready: Any
    _connection: Any

    async def list_pending_session_creations(self, owner, *, cursor=None, limit=32):
        discovery_bounds(cursor, limit)
        expected_owner = owner_key(owner)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT decision_json FROM cayu_session_creation_decisions "
                "WHERE owner_key = %s AND recovery_pending = 1 AND operation_key > %s "
                "ORDER BY operation_key LIMIT %s",
                (expected_owner, cursor or "", limit + 1),
            )
            rows = await cur.fetchall()
            return discovery_page(
                [SessionCreationDecision.model_validate_json(row[0]) for row in rows], limit
            )

    async def _prepare_session_creation_target(self, target, *, authority):
        return await postgres_decision(self, target, authority=authority, mutate=True)

    async def _register_session_creation_target(self, target, *, authority):
        return await postgres_decision(
            self, target, authority=authority, mutate=True, register=True
        )

    async def _exclude_session_creation_target(self, target, *, authority):
        return await postgres_decision(self, target, authority=authority, mutate=True, exclude=True)

    async def _acknowledge_session_creation_settlement(self, target, *, authority):
        return await postgres_decision(
            self, target, authority=authority, mutate=True, acknowledge=True
        )

    async def read_session_creation_decision(self, target):
        target = snapshot_target(target)
        return await exact_read(postgres_decision(self, target))


async def exact_read(operation):
    try:
        result = await operation
    except SessionCreationConflict:
        return ExactConflict()
    except Exception:
        # Failed/corrupt receiving-store readback is not evidence of absence.
        # Caller contract validation occurs before this boundary; cancellation
        # and other control signals deliberately continue to propagate.
        return ExactUnavailable()
    return (
        ExactNotFound() if result is None else ExactMatch[SessionCreationDecision](receipt=result)
    )


async def validate_replay(store, target, session):
    result = await store.read_session_creation_decision(target)
    if not isinstance(result, ExactMatch) or (
        result.receipt.state != "created"
        or result.receipt.session_id != session.id
        or result.receipt.session_instance_id != session.instance_id
    ):
        raise SessionCreationConflict("Creation replay does not match its durable decision.")


def sqlite_read(connection: Any, target: SessionCreationTarget) -> SessionCreationDecision | None:
    row = connection.execute(
        "SELECT decision_json FROM cayu_session_creation_decisions WHERE operation_key = ?",
        (target.key,),
    ).fetchone()
    result = None if row is None else SessionCreationDecision.model_validate_json(row[0])
    if result is not None:
        decide(target, result)
    return result


def sqlite_write(connection: Any, decision: SessionCreationDecision) -> None:
    connection.execute(
        "INSERT INTO cayu_session_creation_decisions (operation_key, owner_key, state, decision_json, recovery_pending) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(operation_key) DO UPDATE SET "
        "state = excluded.state, decision_json = excluded.decision_json, recovery_pending = excluded.recovery_pending",
        (
            decision.target.key,
            owner_key(decision.target.receiving_owner),
            decision.state,
            decision.model_dump_json(),
            int(not decision.settlement_acknowledged),
        ),
    )


async def sqlite_decision(
    store: Any,
    target: SessionCreationTarget,
    *,
    authority: object = None,
    mutate: bool = False,
    exclude: bool = False,
    register: bool = False,
    acknowledge: bool = False,
) -> SessionCreationDecision | None:
    target = snapshot_target(target)
    if mutate:
        require_authority(authority)

    def statement(connection: Any) -> SessionCreationDecision | None:
        if not mutate:
            return sqlite_read(connection, target)
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = decide(
                target,
                sqlite_read(connection, target),
                exclude=exclude,
                register=register,
                acknowledge=acknowledge,
            )
            sqlite_write(connection, result)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await (store._run_write(statement) if mutate else store._run_read(statement))


async def postgres_lock(cur: Any, target: SessionCreationTarget) -> None:
    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        ("session-creation:" + target.key,),
    )


async def postgres_read(cur: Any, target: SessionCreationTarget) -> SessionCreationDecision | None:
    await cur.execute(
        "SELECT decision_json FROM cayu_session_creation_decisions WHERE operation_key = %s",
        (target.key,),
    )
    row = await cur.fetchone()
    result = None if row is None else SessionCreationDecision.model_validate_json(row[0])
    if result is not None:
        decide(target, result)
    return result


async def postgres_write(cur: Any, decision: SessionCreationDecision) -> None:
    await cur.execute(
        "INSERT INTO cayu_session_creation_decisions (operation_key, owner_key, state, decision_json, recovery_pending) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT(operation_key) DO UPDATE SET "
        "state = excluded.state, decision_json = excluded.decision_json, recovery_pending = excluded.recovery_pending",
        (
            decision.target.key,
            owner_key(decision.target.receiving_owner),
            decision.state,
            decision.model_dump_json(),
            int(not decision.settlement_acknowledged),
        ),
    )


async def postgres_decision(
    store: Any,
    target: SessionCreationTarget,
    *,
    authority: object = None,
    mutate: bool = False,
    exclude: bool = False,
    register: bool = False,
    acknowledge: bool = False,
) -> SessionCreationDecision | None:
    target = snapshot_target(target)
    if mutate:
        require_authority(authority)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        if mutate:
            await postgres_lock(cur, target)
        current = await postgres_read(cur, target)
        if not mutate:
            return current
        result = decide(
            target, current, exclude=exclude, register=register, acknowledge=acknowledge
        )
        await postgres_write(cur, result)
        await conn.commit()
        return result


async def memory_decision(
    store: Any,
    target: SessionCreationTarget,
    *,
    authority: object = None,
    mutate: bool = False,
    exclude: bool = False,
    register: bool = False,
    acknowledge: bool = False,
) -> SessionCreationDecision | None:
    target = snapshot_target(target)
    if mutate:
        require_authority(authority)
    async with store._lock:
        current = store._session_creation_decisions.get(target.key)
        if current is not None:
            decide(target, current)
        if not mutate:
            return current
        result = decide(
            target, current, exclude=exclude, register=register, acknowledge=acknowledge
        )
        store._session_creation_decisions[target.key] = result
        return result
