from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from cayu._clock import utc_clock
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_nonblank,
)
from cayu.core.billing import BillingIdentity, copy_billing_identity
from cayu.runtime.budgets import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetLedger,
    BudgetLimit,
    BudgetReconciliation,
    BudgetReconciliationPricing,
    BudgetReservationIdentityConflict,
    BudgetReservationRecord,
    BudgetReservationResult,
    BudgetSettlementCursor,
    BudgetSettlementFallback,
    BudgetSettlementRecord,
    _budget_reservation_amount,
    _budget_settlement_record,
    _copy_budget_settlement_cursor,
    _EffectiveBudgetLimit,
    _ensure_effective_budget_limit,
    _expired_reservation_reason,
    _reconciled_record,
    _reconciliation_from_record,
    _released_record,
    _reservation_is_expired,
    _reservation_result,
    _utc_datetime,
    _validate_amount,
    _validate_reservation_id_batch,
    _validate_reservation_ttl,
    _validate_settlement_page_limit,
    copy_budget_settlement_fallback,
    new_budget_reservation_id,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    copy_model_attempt_identity,
)

from . import _sqlite_support as sqlite_support
from . import migrations as schema

_SQLITE_MIN_REQUIRED_REVISION = 25


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
        self._clock = utc_clock(clock)
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)
        self._connection = sqlite_support.connect(db_path)
        self._connection.row_factory = sqlite3.Row
        sqlite_support.reconcile_schema(
            self._connection,
            schema_mode,
            app_min_supported=_SQLITE_MIN_REQUIRED_REVISION,
        )

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._reservation_ttl_seconds

    async def claim_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                inserted = self._connection.execute(
                    """
                    INSERT OR IGNORE INTO cayu_budget_reservation_identities (
                        reservation_id,
                        publication_session_id,
                        publication_id,
                        published
                    )
                    VALUES (?, ?, ?, 0)
                    """,
                    (reservation_id, publication_session_id, publication_id),
                )
                if inserted.rowcount != 1:
                    existing = self._connection.execute(
                        """
                        SELECT publication_session_id, publication_id
                        FROM cayu_budget_reservation_identities
                        WHERE reservation_id = ?
                        """,
                        (reservation_id,),
                    ).fetchone()
                    if existing is None:
                        raise RuntimeError(
                            "Budget reservation identity claim disappeared during conflict."
                        )
                    if (
                        existing["publication_session_id"],
                        existing["publication_id"],
                    ) != (publication_session_id, publication_id):
                        raise BudgetReservationIdentityConflict(
                            "Budget ledger reused a reservation identity."
                        )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    async def reserve(
        self,
        *,
        reservation_id: str | None = None,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        model_attempt_identity: ModelAttemptIdentity,
        environment_name: str | None = None,
        settlement_event_payload: dict[str, Any] | None = None,
        settlement_fallback: BudgetSettlementFallback | None = None,
        requested_amount: Decimal | None = None,
        billing_identity: BillingIdentity | None = None,
        effective_at: datetime | None = None,
    ) -> BudgetReservationResult:
        reservation_id = (
            new_budget_reservation_id()
            if reservation_id is None
            else require_clean_nonblank(reservation_id, "reservation_id")
        )
        limit = _ensure_effective_budget_limit(
            limit,
            identity_namespace="app_policy",
        )
        session_id = require_clean_nonblank(session_id, "session_id")
        agent_name = require_clean_nonblank(agent_name, "agent_name")
        provider_name = require_clean_nonblank(provider_name, "provider_name")
        model = require_clean_nonblank(model, "model")
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        durable_billing_identity = copy_billing_identity(billing_identity)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._clock()
                durable_settlement_fallback = (
                    BudgetSettlementFallback(
                        settled_at=now,
                        expiration_reason=(
                            None
                            if self._reservation_ttl_seconds is None
                            else _expired_reservation_reason(self._reservation_ttl_seconds)
                        ),
                    )
                    if settlement_fallback is None
                    else copy_budget_settlement_fallback(settlement_fallback)
                )
                pricing_effective_at = (
                    now if effective_at is None else _utc_datetime(effective_at, "effective_at")
                )
                requested = (
                    _budget_reservation_amount(
                        limit=limit,
                        provider_name=provider_name,
                        model=model,
                        effective_at=pricing_effective_at,
                        billing_identity=durable_billing_identity,
                    )
                    if requested_amount is None
                    else _validate_amount(requested_amount, "requested_amount")
                )
                self._reap_expired_unlocked(now, limit=limit)
                current = self._used_amount_unlocked(limit, now=now)
                projected = current + requested
                if projected > limit.max_estimated_cost:
                    # Reaping is an independent terminal transition with its
                    # own outbox evidence. Preserve it even when the new
                    # reservation is rejected.
                    self._connection.commit()
                    return _reservation_result(
                        limit=limit,
                        model_attempt_identity=model_attempt_identity,
                        accepted=False,
                        requested=requested,
                        actual=projected,
                        message=(
                            "Budget reservation failed: "
                            f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                        ),
                    )

                record = BudgetReservationRecord(
                    reservation_id=reservation_id,
                    budget_limit_id=limit.budget_limit_id,
                    model_step_id=model_attempt_identity.model_step_id,
                    model_attempt_id=model_attempt_identity.model_attempt_id,
                    scope=limit.scope,
                    key=limit.key,
                    window=limit.window,
                    currency=limit.currency,
                    session_id=session_id,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    provider_name=provider_name,
                    model=model,
                    billing_identity=durable_billing_identity,
                    settlement_event_payload=settlement_event_payload or {},
                    settlement_fallback=durable_settlement_fallback,
                    reserved_amount=requested,
                    created_at=now,
                    updated_at=now,
                )
                try:
                    self._insert_record_unlocked(record)
                except sqlite3.IntegrityError as exc:
                    if "cayu_budget_reservations.reservation_id" in str(exc):
                        raise BudgetReservationIdentityConflict(
                            "Budget ledger reused a reservation identity."
                        ) from exc
                    raise
                self._connection.commit()
                return _reservation_result(
                    limit=limit,
                    model_attempt_identity=model_attempt_identity,
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

    async def mark_dispatched(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        dispatched_at: datetime | None = None,
    ) -> tuple[BudgetReservationRecord, ...]:
        reservation_ids = _validate_reservation_id_batch(reservation_ids)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        marked_at = (
            sqlite_support.parse_datetime(sqlite_support.format_datetime(dispatched_at))
            if dispatched_at is not None
            else self._clock()
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                records = tuple(
                    self._load_record_unlocked(reservation_id) for reservation_id in reservation_ids
                )
                for record in records:
                    if record.dispatch_id is not None and record.dispatch_id != dispatch_id:
                        raise ValueError(
                            "Budget reservation has a conflicting dispatch: "
                            f"{record.reservation_id}"
                        )
                    if record.dispatch_id is None and record.status != "active":
                        raise ValueError(
                            f"Budget reservation is not active: {record.reservation_id}"
                        )
                dispatched_records = tuple(
                    (
                        record
                        if record.dispatch_id is not None
                        else record.model_copy(
                            update={"dispatch_id": dispatch_id, "dispatched_at": marked_at},
                            deep=True,
                        )
                    )
                    for record in records
                )
                for record in dispatched_records:
                    self._update_record_unlocked(record)
                self._connection.commit()
                return dispatched_records
            except BaseException:
                self._connection.rollback()
                raise

    async def heartbeat(self, *, reservation_id: str) -> bool:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._load_record_unlocked(reservation_id)
                now = self._clock()
                if record.status != "active" or _reservation_is_expired(
                    record,
                    now=now,
                    ttl_seconds=self._reservation_ttl_seconds,
                ):
                    self._connection.commit()
                    return False
                renewed = record.model_copy(update={"updated_at": now}, deep=True)
                self._update_record_unlocked(renewed)
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        settlement_kind: Literal["completed", "conservative"] = "completed",
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
        pricing: BudgetReconciliationPricing | None = None,
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
                    billing_identity=billing_identity,
                )
                reconciliation = _reconciliation_from_record(
                    reconciled,
                    settlement_kind=settlement_kind,
                    pricing=pricing,
                )
                settlement = _budget_settlement_record(record, reconciliation)
                self._insert_or_validate_settlement_unlocked(settlement)
                self._update_record_unlocked(reconciled)
                self._connection.commit()
                return reconciliation
            except Exception:
                self._connection.rollback()
                raise

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = (
            _utc_datetime(occurred_at, "occurred_at") if occurred_at is not None else self._clock()
        )
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                record = self._releasable_record_unlocked(reservation_id)
                released = _released_record(
                    record,
                    reason=reason,
                    updated_at=released_at,
                )
                reconciliation = _reconciliation_from_record(
                    released,
                    settlement_kind="released",
                )
                settlement = _budget_settlement_record(record, reconciliation)
                self._insert_or_validate_settlement_unlocked(settlement)
                self._update_record_unlocked(released)
                self._connection.commit()
                return reconciliation
            except Exception:
                self._connection.rollback()
                raise

    async def load_settlement(self, settlement_id: str) -> BudgetSettlementRecord | None:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE settlement_id = ?
                """,
                (settlement_id,),
            ).fetchone()
            return None if row is None else self._settlement_from_row(row)

    async def list_pending_settlements(
        self,
        *,
        session_id: str | None = None,
        after: BudgetSettlementCursor | None = None,
        limit: int = 100,
    ) -> list[BudgetSettlementRecord]:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        after = _copy_budget_settlement_cursor(after)
        limit = _validate_settlement_page_limit(limit)
        async with self._lock:
            filters = ["event_published = 0"]
            parameters: list[object] = []
            if session_id is not None:
                filters.append("session_id = ?")
                parameters.append(session_id)
            if after is not None:
                formatted_settled_at = sqlite_support.format_datetime(after.settled_at)
                filters.append("(settled_at > ? OR (settled_at = ? AND settlement_id > ?))")
                parameters.extend(
                    [
                        formatted_settled_at,
                        formatted_settled_at,
                        after.settlement_id,
                    ]
                )
            parameters.append(limit)
            rows = self._connection.execute(
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE """
                + " AND ".join(filters)
                + """
                ORDER BY settled_at, settlement_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [self._settlement_from_row(row) for row in rows]

    async def mark_settlement_event_published(
        self,
        *,
        settlement_id: str,
        event_id: str,
    ) -> BudgetSettlementRecord:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT settlement_json, event_published
                    FROM cayu_budget_settlements
                    WHERE settlement_id = ?
                    """,
                    (settlement_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Budget settlement not found: {settlement_id}")
                settlement = self._settlement_from_row(row)
                if settlement.event.id != event_id:
                    raise ValueError(
                        "Budget settlement event acknowledgement has conflicting identity."
                    )
                if not settlement.event_published:
                    self._connection.execute(
                        """
                        UPDATE cayu_budget_settlements
                        SET event_published = 1
                        WHERE settlement_id = ?
                        """,
                        (settlement_id,),
                    )
                    settlement = settlement.model_copy(
                        update={"event_published": True},
                        deep=True,
                    )
                self._connection.commit()
                return settlement
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _used_amount_unlocked(
        self,
        limit: _EffectiveBudgetLimit,
        *,
        now: datetime,
    ) -> Decimal:
        since, until = limit.window.bounds(now=now)
        cutoff = None if since is None else sqlite_support.format_datetime(since)
        upper_cutoff = None if until is None else sqlite_support.format_datetime(until)
        legacy = self._connection.execute(
            """
            SELECT 1
            FROM cayu_budget_reservations
            WHERE budget_limit_id IS NULL
              AND scope = ?
              AND budget_key IS ?
              AND budget_window = ?
              AND currency = ?
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (
                        status = 'reconciled'
                        AND (? IS NULL OR updated_at >= ?)
                        AND (? IS NULL OR updated_at < ?)
                    )
              )
            LIMIT 1
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
        ).fetchone()
        if legacy is not None:
            raise RuntimeError(
                "Budget ledger contains pre-identity reservations for this limit; "
                "exact capacity cannot be verified."
            )
        rows = self._connection.execute(
            """
            SELECT reserved_amount, actual_amount, status
            FROM cayu_budget_reservations
            WHERE budget_limit_id = ?
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (
                        status = 'reconciled'
                        AND (? IS NULL OR updated_at >= ?)
                        AND (? IS NULL OR updated_at < ?)
                    )
              )
            """,
            (
                limit.budget_limit_id,
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

    def _reap_expired_unlocked(
        self,
        now: datetime,
        *,
        limit: _EffectiveBudgetLimit,
    ) -> None:
        if self._reservation_ttl_seconds is None:
            return
        cutoff = now - timedelta(seconds=self._reservation_ttl_seconds)
        rows = self._connection.execute(
            """
            SELECT reservation_id
            FROM cayu_budget_reservations
            WHERE status = 'active'
              AND dispatch_id IS NULL
              AND updated_at <= ?
              AND budget_limit_id = ?
            ORDER BY reservation_id
            """,
            (
                sqlite_support.format_datetime(cutoff),
                limit.budget_limit_id,
            ),
        ).fetchall()
        for row in rows:
            record = self._load_record_unlocked(row["reservation_id"])
            released = _released_record(
                record,
                reason=(
                    record.settlement_fallback.expiration_reason
                    or _expired_reservation_reason(self._reservation_ttl_seconds)
                ),
                updated_at=record.settlement_fallback.settled_at,
            )
            reconciliation = _reconciliation_from_record(
                released,
                settlement_kind="released",
            )
            self._insert_or_validate_settlement_unlocked(
                _budget_settlement_record(record, reconciliation)
            )
            self._update_record_unlocked(released)

    def _insert_record_unlocked(self, record: BudgetReservationRecord) -> None:
        now = sqlite_support.format_datetime(record.created_at)
        updated_at = sqlite_support.format_datetime(record.updated_at)
        self._connection.execute(
            """
            INSERT INTO cayu_budget_reservations (
                reservation_id,
                budget_limit_id,
                model_step_id,
                model_attempt_id,
                scope,
                budget_key,
                budget_window,
                currency,
                session_id,
                agent_name,
                environment_name,
                provider_name,
                model,
                billing_identity_json,
                settlement_event_payload_json,
                settlement_fallback_json,
                dispatch_id,
                dispatched_at,
                reserved_amount,
                actual_amount,
                status,
                reason,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.reservation_id,
                record.budget_limit_id,
                record.model_step_id,
                record.model_attempt_id,
                record.scope,
                record.key,
                record.window.storage_key,
                record.currency,
                record.session_id,
                record.agent_name,
                record.environment_name,
                record.provider_name,
                record.model,
                (
                    None
                    if record.billing_identity is None
                    else record.billing_identity.model_dump_json()
                ),
                sqlite_support.json_dumps(record.settlement_event_payload),
                record.settlement_fallback.model_dump_json(),
                record.dispatch_id,
                (
                    None
                    if record.dispatched_at is None
                    else sqlite_support.format_datetime(record.dispatched_at)
                ),
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
                billing_identity_json = ?,
                dispatch_id = ?,
                dispatched_at = ?,
                status = ?,
                reason = ?,
                updated_at = ?
            WHERE reservation_id = ?
            """,
            (
                None if record.actual_amount is None else str(record.actual_amount),
                (
                    None
                    if record.billing_identity is None
                    else record.billing_identity.model_dump_json()
                ),
                record.dispatch_id,
                (
                    None
                    if record.dispatched_at is None
                    else sqlite_support.format_datetime(record.dispatched_at)
                ),
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
            SELECT reservation_id, budget_limit_id, model_step_id, model_attempt_id,
                   scope, budget_key, budget_window,
                   currency, session_id,
                   agent_name, environment_name, provider_name, model,
                   billing_identity_json, settlement_event_payload_json,
                   settlement_fallback_json, dispatch_id, dispatched_at,
                   reserved_amount, actual_amount,
                   status, reason, created_at, updated_at
            FROM cayu_budget_reservations
            WHERE reservation_id = ?
            """,
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if row["budget_limit_id"] is None:
            raise RuntimeError(
                "Budget reservation predates durable budget-limit identity and "
                "cannot be reconciled safely."
            )
        if row["model_step_id"] is None or row["model_attempt_id"] is None:
            raise RuntimeError(
                "Budget reservation predates durable model-attempt identity and "
                "cannot be reconciled safely."
            )
        return BudgetReservationRecord(
            reservation_id=row["reservation_id"],
            budget_limit_id=row["budget_limit_id"],
            model_step_id=row["model_step_id"],
            model_attempt_id=row["model_attempt_id"],
            scope=row["scope"],
            key=row["budget_key"],
            window=row["budget_window"],
            currency=row["currency"],
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            environment_name=row["environment_name"],
            provider_name=row["provider_name"],
            model=row["model"],
            billing_identity=(
                None
                if row["billing_identity_json"] is None
                else BillingIdentity.model_validate(json.loads(row["billing_identity_json"]))
            ),
            settlement_event_payload=json.loads(row["settlement_event_payload_json"]),
            settlement_fallback=BudgetSettlementFallback.model_validate_json(
                row["settlement_fallback_json"]
            ),
            dispatch_id=row["dispatch_id"],
            dispatched_at=(
                None
                if row["dispatched_at"] is None
                else sqlite_support.parse_datetime(row["dispatched_at"])
            ),
            reserved_amount=Decimal(row["reserved_amount"]),
            actual_amount=(None if row["actual_amount"] is None else Decimal(row["actual_amount"])),
            status=row["status"],
            reason=row["reason"],
            created_at=sqlite_support.parse_datetime(row["created_at"]),
            updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        )

    def _insert_or_validate_settlement_unlocked(
        self,
        settlement: BudgetSettlementRecord,
    ) -> None:
        stored = settlement.model_copy(update={"event_published": False}, deep=True)
        payload = stored.model_dump_json()
        inserted = self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_budget_settlements (
                settlement_id,
                reservation_id,
                session_id,
                settled_at,
                settlement_json,
                event_published
            )
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (
                stored.settlement_id,
                stored.reservation_id,
                stored.session_id,
                sqlite_support.format_datetime(stored.reconciliation.settled_at),
                payload,
            ),
        )
        if inserted.rowcount == 1:
            return
        row = self._connection.execute(
            """
            SELECT settlement_json, event_published
            FROM cayu_budget_settlements
            WHERE settlement_id = ?
            """,
            (stored.settlement_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Budget settlement disappeared during conflict.")
        existing = self._settlement_from_row(row).model_copy(
            update={"event_published": False},
            deep=True,
        )
        if existing != stored:
            raise ValueError(
                f"Budget reservation has a conflicting settlement: {stored.reservation_id}"
            )

    @staticmethod
    def _settlement_from_row(row: sqlite3.Row) -> BudgetSettlementRecord:
        value = BudgetSettlementRecord.model_validate_json(row["settlement_json"])
        return value.model_copy(
            update={"event_published": bool(row["event_published"])},
            deep=True,
        )

    def _active_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    def _releasable_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status == "active" and record.dispatch_id is not None:
            raise ValueError(f"Dispatched budget reservation cannot be released: {reservation_id}")
        if record.status in {"active", "released"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    def _reconcilable_record_unlocked(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._load_record_unlocked(reservation_id)
        if record.status in {"active", "reconciled"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")
