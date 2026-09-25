"""Persistent adapters for native context selection/exclusion arbitration."""

from __future__ import annotations

import json
from typing import Any

from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.sessions._context_selection_fence import (
    CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER,
    ContextViewRetentionEvidence,
    ContextViewSelectionConflict,
    ContextViewSelectionDecision,
    ContextViewSelectionTarget,
    control_transition,
    request_commitment,
    require_authority,
    retention_evidence,
    selected_decision,
    snapshot_request,
)
from cayu.sessions.context_views import (
    ContextViewSelectionReceipt,
    validate_context_view_receipt_storage,
)

_EXCLUSION_COLUMNS = (
    "selection_key, owner_scope, owner_id, owner_incarnation, source_session_id, "
    "source_session_instance_id, request_commitment, decision_json"
)
_SELECTION_COLUMNS = (
    "selection_key, request_commitment, view_id, owner_scope, owner_id, "
    "owner_incarnation, state, pin_commitment, ownership_revision, receipt_json"
)


def exclusion_values(decision):
    request = decision.request
    owner = request.source_owner
    return (
        request.selection_key,
        owner.application_scope,
        owner.owner_id,
        owner.incarnation,
        request.source_session_id,
        request.source_session_instance_id,
        request_commitment(request),
        decision.model_dump_json(),
    )


def reconstruct_exclusion(row):
    if row is None:
        return None
    decision = ContextViewSelectionDecision.model_validate_json(row[7])
    if (
        decision.state not in {"reserved", "excluded"}
        or tuple(row[:7]) != exclusion_values(decision)[:7]
    ):
        raise ValueError("Context selection exclusion indexes conflict with durable evidence.")
    return decision


def reconstruct_decision(request, exclusion_row, selection_row):
    excluded = reconstruct_exclusion(exclusion_row)
    if excluded is not None:
        if selection_row is not None and excluded.state != "reserved":
            raise ValueError("Native selection has contradictory decision evidence.")
        if excluded.request != request:
            raise ContextViewSelectionConflict("Context-view selection request conflicts.")
        if selection_row is None:
            return excluded
    if selection_row is None:
        return None
    receipt = selection_receipt(selection_row)
    return selected_decision(request, receipt, selection_row[1], excluded)


def selection_receipt(row):
    raw = row[9]
    return validate_context_view_receipt_storage(
        ContextViewSelectionReceipt.model_validate(
            json.loads(raw) if isinstance(raw, str) else raw
        ),
        selection_key=row[0],
        view_id=row[2],
        owner_scope=row[3],
        owner_id=row[4],
        owner_incarnation=row[5],
        state=row[6],
        pin_commitment=row[7],
        ownership_revision=row[8],
    )


def reconstruct_retention(target, exclusion_row, selection_row):
    decision = reconstruct_decision(target.request, exclusion_row, selection_row)
    if decision is not None and decision.target != target:
        raise ContextViewSelectionConflict("Retention readback target conflicts.")
    return retention_evidence(
        target, decision, None if selection_row is None else selection_receipt(selection_row)
    )


def sqlite_exclusion(connection, key):
    return connection.execute(
        f"SELECT {_EXCLUSION_COLUMNS} FROM cayu_context_selection_exclusions WHERE selection_key = ?",
        (key,),
    ).fetchone()


def sqlite_decision(connection, request):
    selected = connection.execute(
        f"SELECT {_SELECTION_COLUMNS} FROM cayu_context_view_selections WHERE selection_key = ?",
        (request.selection_key,),
    ).fetchone()
    return reconstruct_decision(
        request, sqlite_exclusion(connection, request.selection_key), selected
    )


def sqlite_snapshot(connection, read):
    with connection:
        connection.execute("BEGIN")
        return read(connection)


async def exact_read(operation, *, schema=ContextViewSelectionDecision):
    try:
        decision = await operation
    except ContextViewSelectionConflict:
        return ExactConflict()
    except Exception:
        return ExactUnavailable()
    return ExactNotFound() if decision is None else ExactMatch[schema](receipt=decision)


class SQLiteContextSelectionFenceMixin:
    _connection: Any
    _context_view_transaction: Any
    _run_read: Any

    async def _read_context_view_retention(self, target):
        target = ContextViewSelectionTarget.model_validate(target)

        def read(connection):
            selected = connection.execute(
                f"SELECT {_SELECTION_COLUMNS} FROM cayu_context_view_selections WHERE selection_key = ?",
                (target.request.selection_key,),
            ).fetchone()
            return reconstruct_retention(
                target, sqlite_exclusion(connection, target.request.selection_key), selected
            )

        return await exact_read(
            self._run_read(lambda connection: sqlite_snapshot(connection, read)),
            schema=ContextViewRetentionEvidence,
        )

    async def read_context_view_selection_decision(self, request):
        request = snapshot_request(request)
        return await exact_read(
            self._run_read(
                lambda connection: sqlite_snapshot(
                    connection, lambda current: sqlite_decision(current, request)
                )
            )
        )

    async def _exclude_context_view_selection(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=True)

    async def _reserve_context_view_selection_control(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=False)

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        require_authority(authority)
        request = snapshot_request(request)
        async with self._context_view_transaction():
            existing = sqlite_decision(self._connection, request)
            decision = control_transition(
                request, existing, exclude=exclude, target=target, register=register
            )
            if decision == existing:
                return decision
            owner = request.source_owner
            count = self._connection.execute(
                "SELECT COUNT(*) FROM cayu_context_selection_exclusions "
                "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            ).fetchone()[0]
            if existing is None and count >= CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER:
                raise OverflowError("Context-view selection control quota exceeded.")
            self._connection.execute(
                f"INSERT INTO cayu_context_selection_exclusions ({_EXCLUSION_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(selection_key) DO UPDATE SET decision_json = excluded.decision_json",
                exclusion_values(decision),
            )
            return decision


async def lock_selection_key(cur, key):
    # The key is globally unique, even for conflicting requests naming different
    # owners/sessions. Acquire it before the existing canonical admission locks.
    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"context-selection-decision:{key}",),
    )


async def postgres_exclusion(cur, key):
    await cur.execute(
        f"SELECT {_EXCLUSION_COLUMNS} FROM cayu_context_selection_exclusions WHERE selection_key = %s",
        (key,),
    )
    return await cur.fetchone()


async def postgres_decision(cur, request):
    await cur.execute(
        f"SELECT {_SELECTION_COLUMNS} FROM cayu_context_view_selections WHERE selection_key = %s",
        (request.selection_key,),
    )
    selected = await cur.fetchone()
    return reconstruct_decision(
        request, await postgres_exclusion(cur, request.selection_key), selected
    )


class PostgresContextSelectionFenceMixin:
    _ensure_ready: Any
    _connection: Any
    _lock_context_view_admission: Any

    async def _read_context_view_retention(self, target):
        target = ContextViewSelectionTarget.model_validate(target)

        async def read():
            await self._ensure_ready()
            async with self._connection() as conn, conn.cursor() as cur:
                await lock_selection_key(cur, target.request.selection_key)
                await cur.execute(
                    f"SELECT {_SELECTION_COLUMNS} FROM cayu_context_view_selections WHERE selection_key = %s",
                    (target.request.selection_key,),
                )
                selected = await cur.fetchone()
                return reconstruct_retention(
                    target, await postgres_exclusion(cur, target.request.selection_key), selected
                )

        return await exact_read(read(), schema=ContextViewRetentionEvidence)

    async def _context_selection_decision(self, request):
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            # Serialize the two-table read with both possible decision writers.
            await lock_selection_key(cur, request.selection_key)
            return await postgres_decision(cur, request)

    async def read_context_view_selection_decision(self, request):
        request = snapshot_request(request)
        return await exact_read(self._context_selection_decision(request))

    async def _exclude_context_view_selection(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=True)

    async def _reserve_context_view_selection_control(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=False)

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        require_authority(authority)
        request = snapshot_request(request)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await lock_selection_key(cur, request.selection_key)
            owner = request.source_owner
            await self._lock_context_view_admission(
                cur, owner=owner, lifecycle=True, session_id=request.source_session_id
            )
            existing = await postgres_decision(cur, request)
            decision = control_transition(
                request, existing, exclude=exclude, target=target, register=register
            )
            if decision == existing:
                return decision
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_context_selection_exclusions "
                "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s",
                (owner.application_scope, owner.owner_id, owner.incarnation),
            )
            if (await cur.fetchone())[
                0
            ] >= CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER and existing is None:
                raise OverflowError("Context-view selection control quota exceeded.")
            await cur.execute(
                f"INSERT INTO cayu_context_selection_exclusions ({_EXCLUSION_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(selection_key) DO UPDATE SET decision_json = excluded.decision_json",
                exclusion_values(decision),
            )
            return decision
