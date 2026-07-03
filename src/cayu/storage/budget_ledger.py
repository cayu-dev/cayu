from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runtime.budgets import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetLedger,
    BudgetLimit,
    BudgetReconciliation,
    BudgetReservationRecord,
    BudgetReservationResult,
    _budget_reservation_amount,
    _clock_or_utc_now,
    _expired_reservation_reason,
    _is_expired_reservation_reason,
    _reconciled_record,
    _reconciliation_from_record,
    _reservation_result,
    _validate_amount,
    _validate_reservation_ttl,
)

from . import _sqlite_support as sqlite_support
from . import migrations as schema


class SQLiteBudgetLedger(BudgetLedger):
    """SQLite-backed atomic budget reservation ledger.

    The ``cayu_budget_reservations`` table is owned by the shared migration
    machinery (ADR 0001 revision 8), not created ad hoc by this class.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: int | None = DEFAULT_RESERVATION_TTL_SECONDS,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteBudgetLedger path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        self._lock = asyncio.Lock()
        self._clock = _clock_or_utc_now(clock)
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)
        self._connection = sqlite_support.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        sqlite_support.reconcile_schema(self._connection, schema_mode)

    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        if type(limit) is not BudgetLimit:
            raise TypeError("limit must be a BudgetLimit.")
        session_id = require_clean_nonblank(session_id, "session_id")
        agent_name = require_clean_nonblank(agent_name, "agent_name")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model = require_clean_nonblank(model, "model")
        requested = _budget_reservation_amount(
            limit=limit,
            provider_name=provider_name,
            model=model,
        )

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._clock()
                self._reap_expired_unlocked(now)
                current = self._used_amount_unlocked(limit, now=now)
                projected = current + requested
                if projected > limit.max_estimated_cost:
                    self._connection.rollback()
                    return _reservation_result(
                        limit=limit,
                        accepted=False,
                        requested=requested,
                        actual=projected,
                        message=(
                            "Budget reservation failed: "
                            f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                        ),
                    )

                record = BudgetReservationRecord(
                    scope=limit.scope,
                    key=limit.key,
                    window=limit.window,
                    currency=limit.currency,
                    session_id=session_id,
                    agent_name=agent_name,
                    provider_name=provider_name,
                    model=model,
                    reserved_amount=requested,
                    created_at=now,
                    updated_at=now,
                )
                self._insert_record_unlocked(record)
                self._connection.commit()
                return _reservation_result(
                    limit=limit,
                    accepted=True,
                    requested=requested,
                    actual=projected,
                    message=(
                        "Budget reserved: "
                        f"{requested} {limit.currency} for {provider_name}/{model}."
                    ),
                    record=record,
                )
            except Exception:
                self._connection.rollback()
                raise

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        reason: str | None = None,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        actual_amount = _validate_amount(actual_amount, "actual_amount")
        reconciled_at = (
            sqlite_support.parse_datetime(sqlite_support.format_datetime(occurred_at))
            if occurred_at is not None
            else self._clock()
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._reconcilable_record_unlocked(reservation_id)
                reconciled = _reconciled_record(
                    record,
                    actual_amount=actual_amount,
                    reason=reason,
                    updated_at=reconciled_at,
                )
                self._update_record_unlocked(reconciled)
                self._connection.commit()
                return _reconciliation_from_record(reconciled)
            except Exception:
                self._connection.rollback()
                raise

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = self._clock()
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._releasable_record_unlocked(reservation_id)
                if record.status == "released":
                    self._connection.commit()
                    return _reconciliation_from_record(record)
                released = record.model_copy(
                    update={
                        "status": "released",
                        "reason": reason,
                        "updated_at": released_at,
                    },
                    deep=True,
                )
                self._update_record_unlocked(released)
                self._connection.commit()
                return _reconciliation_from_record(released)
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _used_amount_unlocked(self, limit: BudgetLimit, *, now: datetime) -> Decimal:
        since, until = limit.window.bounds(now=now)
        cutoff = None if since is None else sqlite_support.format_datetime(since)
        upper_cutoff = None if until is None else sqlite_support.format_datetime(until)
        rows = self._connection.execute(
            """
            SELECT reserved_amount, actual_amount, status
            FROM cayu_budget_reservations
            WHERE scope = ?
              AND budget_key IS ?
              AND budget_window = ?
              AND currency = ?
              AND status IN ('active', 'reconciled')
              AND (? IS NULL OR updated_at >= ?)
              AND (? IS NULL OR updated_at < ?)
            """,
            (
                limit.scope,
                limit.key,
                limit.window.storage_key,
                limit.currency.upper(),
                cutoff,
                cutoff,
                upper_cutoff,
                upper_cutoff,
            ),
        ).fetchall()
        total = Decimal("0")
        for row in rows:
            if row["status"] == "active":
                total += Decimal(row["reserved_amount"])
            elif row["status"] == "reconciled":
                total += Decimal(row["actual_amount"] or "0")
        return total

    def _reap_expired_unlocked(self, now: datetime) -> None:
        if self._reservation_ttl_seconds is None:
            return
        cutoff = now - timedelta(seconds=self._reservation_ttl_seconds)
        self._connection.execute(
            """
            UPDATE cayu_budget_reservations
            SET status = 'released',
                reason = ?,
                updated_at = ?
            WHERE status = 'active'
              AND updated_at <= ?
            """,
            (
                _expired_reservation_reason(self._reservation_ttl_seconds),
                sqlite_support.format_datetime(now),
                sqlite_support.format_datetime(cutoff),
            ),
        )

    def _insert_record_unlocked(self, record: BudgetReservationRecord) -> None:
        now = sqlite_support.format_datetime(record.created_at)
        updated_at = sqlite_support.format_datetime(record.updated_at)
        self._connection.execute(
            """
            INSERT INTO cayu_budget_reservations (
                reservation_id,
                scope,
                budget_key,
                budget_window,
                currency,
                session_id,
                agent_name,
                provider_name,
                model,
                reserved_amount,
                actual_amount,
                status,
                reason,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.reservation_id,
                record.scope,
                record.key,
                record.window.storage_key,
                record.currency,
                record.session_id,
                record.agent_name,
                record.provider_name,
                record.model,
                str(record.reserved_amount),
                None if record.actual_amount is None else str(record.actual_amount),
                record.status,
                record.reason,
                now,
                updated_at,
            ),
        )

    def _update_record_unlocked(self, record: BudgetReservationRecord) -> None:
        updated_at = sqlite_support.format_datetime(record.updated_at)
        cursor = self._connection.execute(
            """
            UPDATE cayu_budget_reservations
            SET actual_amount = ?,
                status = ?,
                reason = ?,
                updated_at = ?
            WHERE reservation_id = ?
            """,
            (
                None if record.actual_amount is None else str(record.actual_amount),
                record.status,
                record.reason,
                updated_at,
                record.reservation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Budget reservation not found: {record.reservation_id}")

    def _load_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        row = self._connection.execute(
            """
            SELECT reservation_id, scope, budget_key, budget_window, currency, session_id,
                   agent_name, provider_name, model, reserved_amount, actual_amount,
                   status, reason, created_at, updated_at
            FROM cayu_budget_reservations
            WHERE reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        return BudgetReservationRecord(
            reservation_id=row["reservation_id"],
            scope=row["scope"],
            key=row["budget_key"],
            window=row["budget_window"],
            currency=row["currency"],
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            provider_name=row["provider_name"],
            model=row["model"],
            reserved_amount=Decimal(row["reserved_amount"]),
            actual_amount=(None if row["actual_amount"] is None else Decimal(row["actual_amount"])),
            status=row["status"],
            reason=row["reason"],
            created_at=sqlite_support.parse_datetime(row["created_at"]),
            updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        )

    def _active_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    def _releasable_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status == "active":
            return record
        if record.status == "released" and _is_expired_reservation_reason(record.reason):
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    def _reconcilable_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status == "active":
            return record
        if record.status == "released" and _is_expired_reservation_reason(record.reason):
            # Reaped by the TTL while still in flight (a long step or a wall-clock jump).
            # Reconcile it anyway so the actual spend is recorded rather than crashing the
            # billed run and silently undercounting the shared budget window.
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")
