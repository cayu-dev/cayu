"""PostgreSQL budget reservations, billing identities and settlement receipts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Literal, LiteralString, cast

from cayu.storage._phase_timing import PostgresTimingScope

try:
    from psycopg.errors import UniqueViolation
    from psycopg_pool import (
        AsyncConnectionPool,  # noqa: TC002 - Preserve runtime constructor annotations.
    )
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets._batch import BudgetBatchResult, batch_failure, prepare_batch, replay_batch
from cayu.budgets.base import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    BudgetBindingAllowanceExhausted,
    BudgetBindingRegistrationConflict,
    BudgetLedger,
    BudgetLimit,
    BudgetReconciliation,
    BudgetReconciliationPricing,
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
from cayu.budgets.billing import BillingIdentity, copy_billing_identity
from cayu.execution_units import ModelAttemptIdentity, copy_model_attempt_identity
from cayu.sessions.base import BudgetReservationIdentityConflict
from cayu.storage import _postgres_base as postgres_base
from cayu.storage import _postgres_support as pg_support
from cayu.storage import migrations as schema


def _budget_advisory_lock_key(limit: _EffectiveBudgetLimit) -> int:
    """Stable 63-bit advisory-lock key for one effective budget limit."""

    material = f"cayu_budget_reservations|{limit.budget_limit_id}"
    digest = sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


class PostgresBudgetLedger(postgres_base._PostgresStoreBase, BudgetLedger):
    """Postgres-backed atomic budget reservation ledger for multi-worker apps.

    ``reserve`` serializes per budget (scope/key/window/currency) under a
    transaction-scoped advisory lock, so concurrent workers on separate
    connections cannot jointly overshoot ``max_estimated_cost``; ``reconcile``
    and ``release`` row-lock the reservation with ``SELECT ... FOR UPDATE``.
    The ``cayu_budget_reservations`` table is owned by the shared migration
    machinery (ADR 0001 revision 8).
    """

    _min_required_revision = 109
    _producer_budget_readback_version = 1

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: int | None = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> None:
        super().__init__(
            conninfo,
            pool=pool,
            min_size=min_size,
            max_size=max_size,
            schema_mode=schema_mode,
        )
        self._clock = utc_clock(clock)
        self._clock_is_injected = clock is not None
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._reservation_ttl_seconds

    async def _require_registered_budget_binding(
        self, *, binding_id: str, authority_digest: str, allowance: int
    ) -> None:
        from cayu.budgets._reservation_scan import (
            binding_read_expectation,
            require_binding_readback,
        )

        binding_id, digest, allowance = binding_read_expectation(
            binding_id, authority_digest, allowance
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT authority_digest, allowance FROM cayu_budget_bindings WHERE binding_id = %s",
                (binding_id,),
            )
            require_binding_readback(await cur.fetchone(), (digest, allowance))

    async def register_budget_binding(
        self,
        *,
        binding_id: str,
        authority_digest: str,
        allowance: int,
    ) -> None:
        binding_id = require_clean_nonblank(binding_id, "binding_id")
        authority_digest = require_clean_nonblank(authority_digest, "authority_digest")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO cayu_budget_bindings
                            (binding_id, authority_digest, allowance, registered_at)
                        VALUES (%s, %s, %s, clock_timestamp())
                        ON CONFLICT (binding_id) DO NOTHING
                        """,
                        (binding_id, authority_digest, allowance),
                    )
                    await cur.execute(
                        "SELECT authority_digest, allowance "
                        "FROM cayu_budget_bindings WHERE binding_id = %s",
                        (binding_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise RuntimeError(
                            "Budget binding registration disappeared during conflict."
                        )
                    if (row[0], row[1]) != (authority_digest, allowance):
                        raise BudgetBindingRegistrationConflict(
                            "Budget binding id is already registered with different authority."
                        )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    @staticmethod
    async def _budget_database_now(cur: Any) -> datetime:
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return authoritative budget time.")
        return pg_support.to_utc(row[0])

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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO cayu_budget_reservation_identities (
                            reservation_id,
                            publication_session_id,
                            publication_id,
                            published
                        )
                        VALUES (%s, %s, %s, FALSE)
                        ON CONFLICT (reservation_id) DO NOTHING
                        RETURNING reservation_id
                        """,
                        (reservation_id, publication_session_id, publication_id),
                    )
                    inserted = await cur.fetchone()
                    if inserted is None:
                        await cur.execute(
                            """
                            SELECT publication_session_id, publication_id
                            FROM cayu_budget_reservation_identities
                            WHERE reservation_id = %s
                            """,
                            (reservation_id,),
                        )
                        existing = await cur.fetchone()
                        if existing is None:
                            raise RuntimeError(
                                "Budget reservation identity claim disappeared during conflict."
                            )
                        if (existing[0], existing[1]) != (
                            publication_session_id,
                            publication_id,
                        ):
                            raise BudgetReservationIdentityConflict(
                                "Budget ledger reused a reservation identity."
                            )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def reserve_batch(
        self,
        *,
        members,
        binding_id=None,
        binding_authority_digest=None,
        binding_allowance=None,
        binding_consumption_id=None,
    ) -> BudgetBatchResult:
        """Reserve all ceilings in one PostgreSQL transaction."""
        members = prepare_batch(members)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    for member in sorted(members, key=lambda item: item.limit.budget_limit_id):
                        await cur.execute(
                            "SELECT pg_advisory_xact_lock(%s)",
                            (_budget_advisory_lock_key(member.limit),),
                        )
                    existing = []
                    for member in members:
                        try:
                            existing.append(
                                await self._load_record(cur, member.record.reservation_id)
                            )
                        except KeyError:
                            existing.append(None)
                    replay = replay_batch(members, tuple(existing))
                    if replay is not None:
                        await conn.commit()
                        return replay
                    now = await self._budget_database_now(cur)
                    records = []
                    projected_usage = []
                    for member in members:
                        await self._reap_expired(cur, now, limit=member.limit)
                        used = await self._used_amount(cur, member.limit, now=now)
                        failure = batch_failure(member, used)
                        if failure is not None:
                            await conn.commit()
                            return BudgetBatchResult((), failure=failure)
                        projected_usage.append(used + member.record.reserved_amount)
                        records.append(
                            member.record.model_copy(
                                update={"created_at": now, "updated_at": now}, deep=True
                            )
                        )
                    if binding_id is not None:
                        if not all(
                            v is not None
                            for v in (
                                binding_authority_digest,
                                binding_allowance,
                                binding_consumption_id,
                            )
                        ):
                            raise ValueError("Complete binding consumption identity is required.")
                        await cur.execute(
                            "SELECT authority_digest, allowance "
                            "FROM cayu_budget_bindings WHERE binding_id = %s FOR UPDATE",
                            (binding_id,),
                        )
                        binding = await cur.fetchone()
                        if binding is None:
                            raise BudgetBindingRegistrationConflict(
                                "Budget binding is not registered exactly."
                            )
                        if (binding[0], binding[1]) != (
                            binding_authority_digest,
                            binding_allowance,
                        ):
                            raise BudgetBindingRegistrationConflict(
                                "Budget binding is not registered exactly."
                            )
                        await cur.execute(
                            """
                            INSERT INTO cayu_budget_binding_consumptions
                                (binding_id, consumption_id, consumed_at)
                            VALUES (%s, %s, clock_timestamp())
                            ON CONFLICT DO NOTHING
                            """,
                            (binding_id, binding_consumption_id),
                        )
                        await cur.execute(
                            "SELECT COUNT(*) FROM cayu_budget_binding_consumptions "
                            "WHERE binding_id = %s",
                            (binding_id,),
                        )
                        count_row = await cur.fetchone()
                        assert count_row is not None
                        assert type(binding_allowance) is int
                        if count_row[0] > binding_allowance:
                            raise BudgetBindingAllowanceExhausted(
                                "Budget binding allowance is exhausted."
                            )
                    for record in records:
                        await self._insert_record(cur, record)
                await conn.commit()
                return BudgetBatchResult(tuple(records), tuple(projected_usage))
            except BaseException:
                await conn.rollback()
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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (_budget_advisory_lock_key(limit),),
                    )
                    reap_now = await self._budget_database_now(cur)
                    await self._reap_expired(cur, reap_now, limit=limit)
                    # Reaping can wait behind a heartbeat or settlement that owns
                    # an existing reservation row.  Stamp the replacement from a
                    # fresh database observation after that wait so a newly
                    # accepted reservation receives its full configured TTL.
                    now = await self._budget_database_now(cur)
                    accounting_now = self._clock() if self._clock_is_injected else now
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
                        accounting_now
                        if effective_at is None
                        else _utc_datetime(effective_at, "effective_at")
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
                    current = await self._used_amount(cur, limit, now=accounting_now)
                    projected = current + requested
                    if projected > limit.max_estimated_cost:
                        # Reaping is an independent terminal transition with
                        # its own outbox evidence. Preserve it even when the
                        # new reservation is rejected.
                        await conn.commit()
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
                        await self._insert_record(cur, record)
                    except UniqueViolation as exc:
                        if (
                            getattr(exc.diag, "constraint_name", None)
                            == "cayu_budget_reservations_pkey"
                        ):
                            raise BudgetReservationIdentityConflict(
                                "Budget ledger reused a reservation identity."
                            ) from exc
                        raise
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return _reservation_result(
            limit=limit,
            model_attempt_identity=model_attempt_identity,
            accepted=True,
            requested=requested,
            actual=projected,
            message=(f"Budget reserved: {requested} {limit.currency} for {provider_name}/{model}."),
            record=record,
        )

    async def mark_dispatched(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        dispatched_at: datetime | None = None,
    ) -> tuple[BudgetReservationRecord, ...]:
        reservation_ids = _validate_reservation_id_batch(reservation_ids)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        supplied_dispatched_at = (
            pg_support.to_utc(dispatched_at) if dispatched_at is not None else None
        )
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    records_by_id = {
                        reservation_id: await self._load_record(
                            cur,
                            reservation_id,
                        )
                        for reservation_id in sorted(reservation_ids)
                    }
                    records = tuple(
                        records_by_id[reservation_id] for reservation_id in reservation_ids
                    )
                    now = await self._budget_database_now(cur)
                    marked_at = now if supplied_dispatched_at is None else supplied_dispatched_at
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
                        if record.dispatch_id is None and _reservation_is_expired(
                            record,
                            now=now,
                            ttl_seconds=self._reservation_ttl_seconds,
                        ):
                            raise ValueError(
                                f"Budget reservation has expired: {record.reservation_id}"
                            )
                    dispatched_records = tuple(
                        (
                            record
                            if record.dispatch_id is not None
                            else record.model_copy(
                                update={
                                    "dispatch_id": dispatch_id,
                                    "dispatched_at": marked_at,
                                },
                                deep=True,
                            )
                        )
                        for record in records
                    )
                    for record in dispatched_records:
                        await self._update_record(cur, record)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return dispatched_records

    async def release_pre_provider_dispatch(
        self,
        *,
        reservation_ids: tuple[str, ...],
        dispatch_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> tuple[BudgetReconciliation, ...]:
        reservation_ids = _validate_reservation_id_batch(reservation_ids)
        dispatch_id = require_clean_nonblank(dispatch_id, "dispatch_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = pg_support.to_utc(occurred_at) if occurred_at is not None else self._clock()
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    records_by_id = {
                        reservation_id: await self._load_record(cur, reservation_id)
                        for reservation_id in sorted(reservation_ids)
                    }
                    records = tuple(
                        records_by_id[reservation_id] for reservation_id in reservation_ids
                    )
                    for record in records:
                        if record.dispatch_id not in {None, dispatch_id}:
                            raise ValueError(
                                "Budget reservation has a conflicting dispatch: "
                                f"{record.reservation_id}"
                            )
                        if record.status not in {"active", "released"}:
                            raise ValueError(
                                f"Budget reservation is not active: {record.reservation_id}"
                            )
                    released_records = tuple(
                        _released_record(record, reason=reason, updated_at=released_at)
                        for record in records
                    )
                    reconciliations = tuple(
                        _reconciliation_from_record(record, settlement_kind="released")
                        for record in released_records
                    )
                    for original, released, reconciliation in zip(
                        records,
                        released_records,
                        reconciliations,
                        strict=True,
                    ):
                        await self._insert_or_validate_settlement(
                            cur,
                            _budget_settlement_record(original, reconciliation),
                        )
                        await self._update_record(cur, released)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return reconciliations

    async def heartbeat(self, *, reservation_id: str) -> bool:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._load_record(cur, reservation_id)
                    now = await self._budget_database_now(cur)
                    if record.status != "active" or _reservation_is_expired(
                        record,
                        now=now,
                        ttl_seconds=self._reservation_ttl_seconds,
                    ):
                        await conn.commit()
                        return False
                    renewed = record.model_copy(update={"updated_at": now}, deep=True)
                    await self._update_record(cur, renewed)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return True

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
        reconciled_at = pg_support.to_utc(occurred_at) if occurred_at is not None else self._clock()
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._reconcilable_record_for_update(cur, reservation_id)
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
                    await self._insert_or_validate_settlement(
                        cur,
                        _budget_settlement_record(record, reconciliation),
                    )
                    await self._update_record(cur, reconciled)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return reconciliation

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
        occurred_at: datetime | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        released_at = pg_support.to_utc(occurred_at) if occurred_at is not None else self._clock()
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    record = await self._releasable_record_for_update(cur, reservation_id)
                    released = _released_record(
                        record,
                        reason=reason,
                        updated_at=released_at,
                    )
                    reconciliation = _reconciliation_from_record(
                        released,
                        settlement_kind="released",
                    )
                    await self._insert_or_validate_settlement(
                        cur,
                        _budget_settlement_record(record, reconciliation),
                    )
                    await self._update_record(cur, released)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return reconciliation

    async def load_settlement(self, settlement_id: str) -> BudgetSettlementRecord | None:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE settlement_id = %s
                """,
                (settlement_id,),
            )
            row = await cur.fetchone()
            return None if row is None else self._settlement_from_row(row)

    async def load_reservation(
        self,
        reservation_id: str,
    ) -> BudgetReservationRecord | None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            try:
                record = await self._load_record(cur, reservation_id, for_update=False)
            except KeyError:
                return None
            return record.model_copy(deep=True)

    async def _scan_reservation_records(
        self, *, session_id: str, after: str | None = None, limit: int = 128
    ) -> tuple[BudgetReservationRecord, ...]:
        from cayu.budgets._reservation_scan import reservation_scan_bounds

        session_id = require_clean_nonblank(session_id, "session_id")
        after, limit = reservation_scan_bounds(after, limit)
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute(
                "SELECT reservation_id FROM cayu_budget_reservations "
                "WHERE session_id = %s AND reservation_id > %s ORDER BY reservation_id LIMIT %s",
                (session_id, after or "", limit),
            )
            rows = await cur.fetchall()
            return tuple([await self._load_record(cur, row[0], for_update=False) for row in rows])

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
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
            filters = ["NOT event_published"]
            parameters: list[object] = []
            if session_id is not None:
                filters.append("session_id = %s")
                parameters.append(session_id)
            if after is not None:
                filters.append("(settled_at > %s OR (settled_at = %s AND settlement_id > %s))")
                parameters.extend(
                    [
                        pg_support.to_utc(after.settled_at),
                        pg_support.to_utc(after.settled_at),
                        after.settlement_id,
                    ]
                )
            parameters.append(limit)
            query = (
                """
                SELECT settlement_json, event_published
                FROM cayu_budget_settlements
                WHERE """
                + " AND ".join(filters)
                + """
                ORDER BY settled_at, settlement_id
                LIMIT %s
                """
            )
            await cur.execute(
                cast("LiteralString", query),
                parameters,
            )
            return [self._settlement_from_row(row) for row in await cur.fetchall()]

    async def mark_settlement_event_published(
        self,
        *,
        settlement_id: str,
        event_id: str,
    ) -> BudgetSettlementRecord:
        settlement_id = require_clean_nonblank(settlement_id, "settlement_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        await self._ensure_ready()
        async with PostgresTimingScope(self._pool.connection()) as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT settlement_json, event_published
                        FROM cayu_budget_settlements
                        WHERE settlement_id = %s
                        FOR UPDATE
                        """,
                        (settlement_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise KeyError(f"Budget settlement not found: {settlement_id}")
                    settlement = self._settlement_from_row(row)
                    if settlement.event.id != event_id:
                        raise ValueError(
                            "Budget settlement event acknowledgement has conflicting identity."
                        )
                    if not settlement.event_published:
                        await cur.execute(
                            """
                            UPDATE cayu_budget_settlements
                            SET event_published = TRUE
                            WHERE settlement_id = %s
                            """,
                            (settlement_id,),
                        )
                        settlement = settlement.model_copy(
                            update={"event_published": True},
                            deep=True,
                        )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return settlement

    async def _reap_expired(
        self,
        cur: Any,
        now: datetime,
        *,
        limit: _EffectiveBudgetLimit,
    ) -> None:
        if self._reservation_ttl_seconds is None:
            return
        cutoff = utc_duration_cutoff(now, self._reservation_ttl_seconds)
        if cutoff is None:
            return
        await cur.execute(
            """
            SELECT reservation_id
            FROM cayu_budget_reservations
            WHERE status = 'active'
              AND dispatch_id IS NULL
              AND updated_at <= %s
              AND budget_limit_id = %s
            ORDER BY reservation_id
            FOR UPDATE
            """,
            (
                pg_support.to_utc(cutoff),
                limit.budget_limit_id,
            ),
        )
        for row in await cur.fetchall():
            record = await self._load_record(cur, row[0])
            released = _released_record(
                record,
                reason=(
                    record.settlement_fallback.expiration_reason
                    or _expired_reservation_reason(self._reservation_ttl_seconds)
                ),
                updated_at=now,
            )
            reconciliation = _reconciliation_from_record(
                released,
                settlement_kind="released",
            )
            await self._insert_or_validate_settlement(
                cur,
                _budget_settlement_record(record, reconciliation),
            )
            await self._update_record(cur, released)

    async def _used_amount(
        self,
        cur: Any,
        limit: _EffectiveBudgetLimit,
        *,
        now: datetime,
    ) -> Decimal:
        since, until = limit.window.bounds(now=now)
        reconciled_bound_sql = ""
        params: list[object] = [
            limit.budget_limit_id,
        ]
        if since is not None:
            reconciled_bound_sql += " AND updated_at >= %s"
            params.append(pg_support.to_utc(since))
        if until is not None:
            reconciled_bound_sql += " AND updated_at < %s"
            params.append(pg_support.to_utc(until))
        legacy_params: list[object] = [
            limit.scope,
            limit.key,
            limit.window.storage_key,
            limit.currency.upper(),
        ]
        legacy_bound_sql = ""
        if since is not None:
            legacy_bound_sql += " AND updated_at >= %s"
            legacy_params.append(pg_support.to_utc(since))
        if until is not None:
            legacy_bound_sql += " AND updated_at < %s"
            legacy_params.append(pg_support.to_utc(until))
        await cur.execute(
            f"""
            SELECT 1
            FROM cayu_budget_reservations
            WHERE budget_limit_id IS NULL
              AND scope = %s
              AND budget_key IS NOT DISTINCT FROM %s
              AND budget_window = %s
              AND currency = %s
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (status = 'reconciled' {legacy_bound_sql})
              )
            LIMIT 1
            """,
            legacy_params,
        )
        if await cur.fetchone() is not None:
            raise RuntimeError(
                "Budget ledger contains pre-identity reservations for this limit; "
                "exact capacity cannot be verified."
            )
        await cur.execute(
            f"""
            SELECT reserved_amount, actual_amount, status
            FROM cayu_budget_reservations
            WHERE budget_limit_id = %s
              AND status IN ('active', 'reconciled')
              AND (
                    status = 'active'
                    OR (status = 'reconciled' {reconciled_bound_sql})
              )
            """,
            params,
        )
        total = Decimal("0")
        for row in await cur.fetchall():
            if row[2] == "active":
                total += row[0]
            elif row[2] == "reconciled":
                total += Decimal("0") if row[1] is None else row[1]
        return total

    async def _insert_record(self, cur: Any, record: BudgetReservationRecord) -> None:
        await cur.execute(
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
                billing_identity,
                settlement_event_payload,
                settlement_fallback,
                dispatch_id,
                dispatched_at,
                reserved_amount,
                actual_amount,
                status,
                reason,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
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
                    else pg_support._dumps(record.billing_identity.model_dump(mode="json"))
                ),
                pg_support._dumps(record.settlement_event_payload),
                pg_support._dumps(record.settlement_fallback.model_dump(mode="json")),
                record.dispatch_id,
                pg_support.to_utc_optional(record.dispatched_at),
                record.reserved_amount,
                record.actual_amount,
                record.status,
                record.reason,
                pg_support.to_utc(record.created_at),
                pg_support.to_utc(record.updated_at),
            ),
        )

    async def _update_record(self, cur: Any, record: BudgetReservationRecord) -> None:
        await cur.execute(
            """
            UPDATE cayu_budget_reservations
            SET actual_amount = %s,
                billing_identity = %s,
                dispatch_id = %s,
                dispatched_at = %s,
                status = %s,
                reason = %s,
                updated_at = %s
            WHERE reservation_id = %s
            """,
            (
                record.actual_amount,
                (
                    None
                    if record.billing_identity is None
                    else pg_support._dumps(record.billing_identity.model_dump(mode="json"))
                ),
                record.dispatch_id,
                pg_support.to_utc_optional(record.dispatched_at),
                record.status,
                record.reason,
                pg_support.to_utc(record.updated_at),
                record.reservation_id,
            ),
        )
        if cur.rowcount != 1:
            raise KeyError(f"Budget reservation not found: {record.reservation_id}")

    async def _load_record(
        self,
        cur: Any,
        reservation_id: str,
        *,
        for_update: bool = True,
    ) -> BudgetReservationRecord:
        query = """
            SELECT reservation_id, budget_limit_id, model_step_id, model_attempt_id,
                   scope, budget_key, budget_window,
                   currency, session_id,
                   agent_name, environment_name, provider_name, model,
                   billing_identity, settlement_event_payload, settlement_fallback,
                   dispatch_id, dispatched_at,
                   reserved_amount, actual_amount,
                   status, reason, created_at, updated_at
            FROM cayu_budget_reservations
            WHERE reservation_id = %s
            """
        if for_update:
            query += " FOR UPDATE"
        await cur.execute(query, (reservation_id,))
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if row[1] is None:
            raise RuntimeError(
                "Budget reservation predates durable budget-limit identity and "
                "cannot be reconciled safely."
            )
        if row[2] is None or row[3] is None:
            raise RuntimeError(
                "Budget reservation predates durable model-attempt identity and "
                "cannot be reconciled safely."
            )
        return BudgetReservationRecord(
            reservation_id=row[0],
            budget_limit_id=row[1],
            model_step_id=row[2],
            model_attempt_id=row[3],
            scope=row[4],
            key=row[5],
            window=row[6],
            currency=row[7],
            session_id=row[8],
            agent_name=row[9],
            environment_name=row[10],
            provider_name=row[11],
            model=row[12],
            billing_identity=(
                None
                if row[13] is None
                else BillingIdentity.model_validate(pg_support._json_obj(row[13]))
            ),
            settlement_event_payload=pg_support._json_obj(row[14]),
            settlement_fallback=BudgetSettlementFallback.model_validate(
                pg_support._json_obj(row[15])
            ),
            dispatch_id=row[16],
            dispatched_at=pg_support.to_utc_optional(row[17]),
            reserved_amount=row[18],
            actual_amount=row[19],
            status=row[20],
            reason=row[21],
            created_at=pg_support.to_utc(row[22]),
            updated_at=pg_support.to_utc(row[23]),
        )

    async def _insert_or_validate_settlement(
        self,
        cur: Any,
        settlement: BudgetSettlementRecord,
    ) -> None:
        stored = settlement.model_copy(update={"event_published": False}, deep=True)
        await cur.execute(
            """
            INSERT INTO cayu_budget_settlements (
                settlement_id,
                reservation_id,
                session_id,
                settled_at,
                settlement_json,
                event_published
            )
            VALUES (%s, %s, %s, %s, %s, FALSE)
            ON CONFLICT (settlement_id) DO NOTHING
            RETURNING settlement_id
            """,
            (
                stored.settlement_id,
                stored.reservation_id,
                stored.session_id,
                pg_support.to_utc(stored.reconciliation.settled_at),
                pg_support._dumps(stored.model_dump(mode="json")),
            ),
        )
        if await cur.fetchone() is not None:
            return
        await cur.execute(
            """
            SELECT settlement_json, event_published
            FROM cayu_budget_settlements
            WHERE settlement_id = %s
            """,
            (stored.settlement_id,),
        )
        row = await cur.fetchone()
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
    def _settlement_from_row(row: Any) -> BudgetSettlementRecord:
        settlement = BudgetSettlementRecord.model_validate(pg_support._json_obj(row[0]))
        return settlement.model_copy(
            update={"event_published": bool(row[1])},
            deep=True,
        )

    async def _active_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record(cur, reservation_id)
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    async def _releasable_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record(cur, reservation_id)
        if record.status == "active" and record.dispatch_id is not None:
            raise ValueError(f"Dispatched budget reservation cannot be released: {reservation_id}")
        if record.status in {"active", "released"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    async def _reconcilable_record_for_update(
        self,
        cur: Any,
        reservation_id: str,
    ) -> BudgetReservationRecord:
        record = await self._load_record(cur, reservation_id)
        if record.status in {"active", "reconciled"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")
