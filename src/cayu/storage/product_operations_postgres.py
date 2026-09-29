"""PostgreSQL product operation store for the maintained public-agent service."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

try:
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "Cayu's Postgres stores require the optional psycopg packages. "
        'Install them with `pip install "cayu[postgres]"`.'
    ) from exc

from cayu._clock import utc_clock
from cayu.server import (
    ProductExecutionClaimLost,
    ProductOperation,
    ProductOperationExecutionClaim,
    ProductOperationReservation,
    ProductRecoveryStatus,
    ProductResultReceipt,
    ServiceIdentityStoreKind,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _product_operations as rules
from cayu.storage import migrations as schema
from cayu.storage._product_operation_schema import PRODUCT_OPERATIONS_TABLE
from cayu.storage.postgres import _PostgresStoreBase

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

_LOCK_OPERATION = f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = %s FOR UPDATE"


class PostgresProductOperationStore(_PostgresStoreBase):
    """Tenant/resource authority shared by product service processes on PostgreSQL.

    This store, not Cayu labels, metadata, or runtime identifiers, is the product
    authorization boundary. Product reads are tenant-qualified by public id; the
    private session-id lookup exists only for trusted continuation hooks.

    Every claim, heartbeat, receipt, and settlement locks the one operation row
    and evaluates lease expiry against the database clock after that lock, so
    service processes on different hosts agree on ownership. Terminal updates
    are conditional on the current claim, and a completed result requires the
    content-bound publication receipt written before Cayu session completion.

    ``cayu_product_operations`` joins Cayu's shared revision history; like the
    other Postgres stores, the default ``schema_mode`` is ``VALIDATE`` and
    ``cayu storage migrate`` creates or upgrades the table. ``clock`` replaces
    the database lease clock for tests.
    """

    category: ServiceIdentityStoreKind = ServiceIdentityStoreKind.DURABLE

    _min_required_revision = rules.PRODUCT_OPERATION_MIN_REQUIRED_REVISION

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        clock: Callable[[], datetime] | None = None,
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

    async def _lease_now(self, cur: Any) -> datetime:
        if self._clock_is_injected:
            return self._clock()
        # Evaluate after the row lock: a transaction timestamp taken before a
        # lock wait could make an expired claim look live under contention.
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return authoritative lease time.")
        return pg_support.to_utc(row["clock_timestamp"])

    async def reserve(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation:
        requested = rules.pending_operation(
            tenant_id=tenant_id,
            subject_id=subject_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            public_id=public_id,
            work_id=work_id,
            session_id=session_id,
            task_id=task_id,
            request_text=request_text,
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            # A concurrent reservation of the same key waits for the first
            # transaction; this statement then observes its committed row.
            await cur.execute(
                f"""INSERT INTO {PRODUCT_OPERATIONS_TABLE} (
                    work_id, public_id, tenant_id, subject_id, idempotency_key,
                    request_fingerprint, session_id, task_id, request_text, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING work_id""",
                (
                    requested.work_id,
                    requested.public_id,
                    requested.tenant_id,
                    requested.subject_id,
                    requested.idempotency_key,
                    requested.request_fingerprint,
                    requested.session_id,
                    requested.task_id,
                    requested.request_text,
                ),
            )
            if await cur.fetchone() is not None:
                await conn.commit()
                return ProductOperationReservation(operation=requested, created=True)
            await cur.execute(
                f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE idempotency_key = %s",
                (requested.idempotency_key,),
            )
            row = await cur.fetchone()
            await conn.commit()
        if row is None:
            raise RuntimeError("Product idempotency reservation disappeared.")
        return rules.existing_reservation(
            rules.operation_from_row(row),
            tenant_id=requested.tenant_id,
            request_fingerprint=requested.request_fingerprint,
        )

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
        tenant_key = rules.lookup_identity(tenant_id, "tenant_id")
        public_key = rules.lookup_identity(public_id, "public_id")
        if tenant_key is None or public_key is None:
            return None
        return await self._find(
            f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE tenant_id = %s AND public_id = %s",
            (tenant_key, public_key),
        )

    async def find_by_session_id(self, *, session_id: str) -> ProductOperation | None:
        session_key = rules.lookup_identity(session_id, "session_id")
        if session_key is None:
            return None
        return await self._find(
            f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE session_id = %s",
            (session_key,),
        )

    async def _find(self, sql: str, parameters: tuple[str, ...]) -> ProductOperation | None:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, parameters)
            row = await cur.fetchone()
            await conn.commit()
        return None if row is None else rules.operation_from_row(row)

    async def claim_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        lease = timedelta(seconds=rules.lease_seconds(lease_seconds))
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_LOCK_OPERATION, (work_id,))
            row = await cur.fetchone()
            if row is None:
                await conn.commit()
                return None
            operation = rules.operation_from_row(row)
            if operation.status != "pending":
                await conn.commit()
                return ProductOperationExecutionClaim(operation=operation, acquired=False)
            now = await self._lease_now(cur)
            expires_at = pg_support.to_utc_optional(row["execution_claim_expires_at"])
            if not rules.claim_available(
                current_claim_id=row["execution_claim_id"],
                claim_id=claim_id,
                lease_expired=expires_at is None or expires_at <= now,
            ):
                await conn.commit()
                return ProductOperationExecutionClaim(operation=operation, acquired=False)
            renewed = now + lease if expires_at is None else max(expires_at, now + lease)
            await cur.execute(
                f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                   SET execution_claim_id = %s, execution_claim_expires_at = %s
                   WHERE work_id = %s AND status = 'pending'""",
                (claim_id, renewed, work_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Product execution claim changed during atomic claim.")
            await conn.commit()
        return ProductOperationExecutionClaim(operation=operation, acquired=True)

    async def heartbeat_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        lease = timedelta(seconds=rules.lease_seconds(lease_seconds))
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT status, execution_claim_id, execution_claim_expires_at "
                f"FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = %s FOR UPDATE",
                (work_id,),
            )
            row = await cur.fetchone()
            if row is None:
                await conn.commit()
                return False
            if row["status"] == "pending" and row["execution_claim_id"] == claim_id:
                now = await self._lease_now(cur)
                expires_at = pg_support.to_utc_optional(row["execution_claim_expires_at"])
                # Concurrent heartbeats for one claim never shorten its lease.
                renewed = now + lease if expires_at is None else max(expires_at, now + lease)
                await cur.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET execution_claim_expires_at = %s
                       WHERE work_id = %s AND status = 'pending' AND execution_claim_id = %s""",
                    (renewed, work_id, claim_id),
                )
                if cur.rowcount != 1:
                    raise RuntimeError("Product execution claim changed during heartbeat.")
                await conn.commit()
                return True
            await conn.commit()
        return rules.heartbeat_recognizes_settlement(
            status=row["status"],
            current_claim_id=row["execution_claim_id"],
            claim_id=claim_id,
        )

    async def release_execution(self, *, work_id: str, claim_id: str) -> bool:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT status, execution_claim_id "
                f"FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = %s FOR UPDATE",
                (work_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during execution-claim release.")
            outcome = rules.release_outcome(
                status=row["status"],
                current_claim_id=row["execution_claim_id"],
                claim_id=claim_id,
            )
            if outcome is not None:
                await conn.commit()
                return outcome
            await cur.execute(
                f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                   SET execution_claim_id = NULL, execution_claim_expires_at = NULL
                   WHERE work_id = %s AND status = 'pending' AND execution_claim_id = %s""",
                (work_id, claim_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Product execution claim changed during atomic release.")
            await conn.commit()
        return True

    async def record_result_receipt(
        self,
        *,
        work_id: str,
        claim_id: str,
        receipt: ProductResultReceipt,
    ) -> ProductResultReceipt:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        receipt = rules.result_receipt(receipt)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_LOCK_OPERATION, (work_id,))
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during result publication.")
            if not rules.receipt_write_required(
                rules.operation_from_row(row),
                current_claim_id=row["execution_claim_id"],
                claim_id=claim_id,
                receipt=receipt,
            ):
                await conn.commit()
                return receipt
            await cur.execute(
                f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                   SET result_receipt = %s::jsonb, recovery_status = NULL
                   WHERE work_id = %s AND status = 'pending' AND execution_claim_id = %s""",
                (rules.encode_receipt(receipt), work_id, claim_id),
            )
            if cur.rowcount != 1:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before result publication."
                )
            await conn.commit()
        return receipt

    async def record_recovery_status(
        self,
        *,
        work_id: str,
        claim_id: str,
        recovery_status: ProductRecoveryStatus,
    ) -> ProductOperation:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        recovery_status = rules.recovery_status(recovery_status)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_LOCK_OPERATION, (work_id,))
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during recovery reporting.")
            rules.require_recovery_owner(
                rules.operation_from_row(row),
                current_claim_id=row["execution_claim_id"],
                claim_id=claim_id,
            )
            await cur.execute(
                f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                   SET recovery_status = %s
                   WHERE work_id = %s AND status = 'pending' AND execution_claim_id = %s
                   RETURNING *""",
                (recovery_status, work_id, claim_id),
            )
            updated = await cur.fetchone()
            if updated is None:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before recovery reporting."
                )
            await conn.commit()
        return rules.operation_from_row(updated)

    async def finish(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: Literal["completed", "failed"],
        result: str | None,
    ) -> ProductOperation:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        status, result = rules.settlement(status, result)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_LOCK_OPERATION, (work_id,))
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during completion.")
            operation = rules.operation_from_row(row)
            if not rules.finish_write_required(
                operation,
                current_claim_id=row["execution_claim_id"],
                claim_id=claim_id,
                status=status,
                result=result,
            ):
                await conn.commit()
                return operation
            # Terminal rows retain the settling claim so a repeated write can
            # reconstruct this result after acknowledgement loss.
            await cur.execute(
                f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                   SET status = %s, result = %s, recovery_status = NULL,
                       execution_claim_expires_at = NULL
                   WHERE work_id = %s AND status = 'pending' AND execution_claim_id = %s
                   RETURNING *""",
                (status, result, work_id, claim_id),
            )
            updated = await cur.fetchone()
            if updated is None:
                raise ProductExecutionClaimLost(
                    "Product execution ownership was lost before completion."
                )
            await conn.commit()
        return rules.operation_from_row(updated)


__all__ = ["PostgresProductOperationStore"]
