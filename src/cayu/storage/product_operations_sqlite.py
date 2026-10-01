"""SQLite product operation store for the maintained public-agent service."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

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
from cayu.storage import _product_operations as rules
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._product_operation_schema import PRODUCT_OPERATIONS_TABLE
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage.targets import require_sqlite_store_allowed

_T = TypeVar("_T")

_SELECT_OPERATION = f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = ?"


class SQLiteProductOperationStore:
    """Tenant/resource authority for a maintained product service on SQLite.

    This store, not Cayu labels, metadata, or runtime identifiers, is the product
    authorization boundary. Product reads are tenant-qualified by public id; the
    private session-id lookup exists only for trusted continuation hooks.

    Every mutation runs in one ``BEGIN IMMEDIATE`` transaction. Execution claims
    use store-owned lease time, terminal updates are conditional on the current
    claim, and a completed result requires the content-bound publication receipt
    written before Cayu session completion. SQLite serializes writers on one
    host; use :class:`PostgresProductOperationStore` for service processes that
    do not share one filesystem.

    The table joins Cayu's shared schema revision history. ``schema_mode``
    follows the other SQLite stores: ``CREATE`` initializes an empty database and
    otherwise validates; an existing older database requires
    ``cayu storage migrate``. ``clock`` replaces the lease clock for tests.
    """

    category: ServiceIdentityStoreKind = ServiceIdentityStoreKind.DURABLE

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        require_sqlite_store_allowed("SQLiteProductOperationStore")
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteProductOperationStore path must be a nonblank string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self.category = (
            ServiceIdentityStoreKind.DEVELOPMENT
            if str(db_path) == ":memory:"
            else ServiceIdentityStoreKind.DURABLE
        )
        self._clock = utc_clock(clock)
        self._lock = TimedStoreLock()
        self._closed = False
        self._connection = sqlite_support.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=rules.PRODUCT_OPERATION_MIN_REQUIRED_REVISION,
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
            raise RuntimeError("SQLiteProductOperationStore is closed.")
        return await _run_off_thread_with_connection_ownership(
            self._lock, self._connection, operation
        )

    def _now_ms(self) -> int:
        return int(self._clock().timestamp() * 1000)

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

        def reserve(connection: sqlite3.Connection) -> ProductOperationReservation:
            with sqlite_support._transaction(connection):
                row = connection.execute(
                    f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE idempotency_key = ?",
                    (requested.idempotency_key,),
                ).fetchone()
                if row is not None:
                    return rules.existing_reservation(
                        rules.operation_from_row(row),
                        tenant_id=requested.tenant_id,
                        request_fingerprint=requested.request_fingerprint,
                    )
                connection.execute(
                    f"""INSERT INTO {PRODUCT_OPERATIONS_TABLE} (
                        work_id, public_id, tenant_id, subject_id, idempotency_key,
                        request_fingerprint, session_id, task_id, request_text, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
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
            return ProductOperationReservation(operation=requested, created=True)

        return await self._run(reserve)

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
        tenant_key = rules.lookup_identity(tenant_id, "tenant_id")
        public_key = rules.lookup_identity(public_id, "public_id")
        if tenant_key is None or public_key is None:
            return None
        return await self._find(
            f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE tenant_id = ? AND public_id = ?",
            (tenant_key, public_key),
        )

    async def find_by_session_id(self, *, session_id: str) -> ProductOperation | None:
        session_key = rules.lookup_identity(session_id, "session_id")
        if session_key is None:
            return None
        return await self._find(
            f"SELECT * FROM {PRODUCT_OPERATIONS_TABLE} WHERE session_id = ?",
            (session_key,),
        )

    async def _find(self, sql: str, parameters: tuple[str, ...]) -> ProductOperation | None:
        def find(connection: sqlite3.Connection) -> ProductOperation | None:
            with sqlite_support._transaction(connection, begin_immediate=False):
                row = connection.execute(sql, parameters).fetchone()
            return None if row is None else rules.operation_from_row(row)

        return await self._run(find)

    async def claim_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        lease_ms = rules.lease_seconds(lease_seconds) * 1000

        def claim(connection: sqlite3.Connection) -> ProductOperationExecutionClaim | None:
            with sqlite_support._transaction(connection):
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
                if row is None:
                    return None
                operation = rules.operation_from_row(row)
                if operation.status != "pending":
                    return ProductOperationExecutionClaim(operation=operation, acquired=False)
                now = self._now_ms()
                expires_at = row["execution_claim_expires_at"]
                if not rules.claim_available(
                    current_claim_id=row["execution_claim_id"],
                    claim_id=claim_id,
                    lease_expired=expires_at is None or expires_at <= now,
                ):
                    return ProductOperationExecutionClaim(operation=operation, acquired=False)
                changed = connection.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET execution_claim_id = ?, execution_claim_expires_at = ?
                       WHERE work_id = ? AND status = 'pending'
                         AND execution_claim_id IS ?""",
                    (
                        claim_id,
                        max(expires_at or 0, now + lease_ms),
                        work_id,
                        row["execution_claim_id"],
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("Product execution claim changed during atomic claim.")
            return ProductOperationExecutionClaim(operation=operation, acquired=True)

        return await self._run(claim)

    async def heartbeat_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")
        lease_ms = rules.lease_seconds(lease_seconds) * 1000

        def heartbeat(connection: sqlite3.Connection) -> bool:
            with sqlite_support._transaction(connection):
                row = connection.execute(
                    "SELECT status, execution_claim_id, execution_claim_expires_at "
                    f"FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = ?",
                    (work_id,),
                ).fetchone()
                if row is None:
                    return False
                if row["status"] == "pending" and row["execution_claim_id"] == claim_id:
                    # Concurrent heartbeats for one claim never shorten its lease.
                    connection.execute(
                        f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                           SET execution_claim_expires_at = ?
                           WHERE work_id = ? AND status = 'pending'
                             AND execution_claim_id = ?""",
                        (
                            max(row["execution_claim_expires_at"] or 0, self._now_ms() + lease_ms),
                            work_id,
                            claim_id,
                        ),
                    )
                    return True
                return rules.heartbeat_recognizes_settlement(
                    status=row["status"],
                    current_claim_id=row["execution_claim_id"],
                    claim_id=claim_id,
                )

        return await self._run(heartbeat)

    async def release_execution(self, *, work_id: str, claim_id: str) -> bool:
        work_id = rules.identity(work_id, "work_id")
        claim_id = rules.identity(claim_id, "claim_id")

        def release(connection: sqlite3.Connection) -> bool:
            with sqlite_support._transaction(connection):
                row = connection.execute(
                    "SELECT status, execution_claim_id "
                    f"FROM {PRODUCT_OPERATIONS_TABLE} WHERE work_id = ?",
                    (work_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Product work disappeared during execution-claim release.")
                outcome = rules.release_outcome(
                    status=row["status"],
                    current_claim_id=row["execution_claim_id"],
                    claim_id=claim_id,
                )
                if outcome is not None:
                    return outcome
                changed = connection.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET execution_claim_id = NULL, execution_claim_expires_at = NULL
                       WHERE work_id = ? AND status = 'pending' AND execution_claim_id = ?""",
                    (work_id, claim_id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("Product execution claim changed during atomic release.")
                return True

        return await self._run(release)

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
        encoded = rules.encode_receipt(receipt)

        def record(connection: sqlite3.Connection) -> ProductResultReceipt:
            with sqlite_support._transaction(connection):
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
                if row is None:
                    raise RuntimeError("Product work disappeared during result publication.")
                if not rules.receipt_write_required(
                    rules.operation_from_row(row),
                    current_claim_id=row["execution_claim_id"],
                    claim_id=claim_id,
                    receipt=receipt,
                ):
                    return receipt
                changed = connection.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET result_receipt = ?, recovery_status = NULL
                       WHERE work_id = ? AND status = 'pending'
                         AND result_receipt IS ? AND execution_claim_id = ?""",
                    (encoded, work_id, row["result_receipt"], claim_id),
                ).rowcount
                if changed != 1:
                    raise ProductExecutionClaimLost(
                        "Product execution ownership was lost before result publication."
                    )
            return receipt

        return await self._run(record)

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

        def record(connection: sqlite3.Connection) -> ProductOperation:
            with sqlite_support._transaction(connection):
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
                if row is None:
                    raise RuntimeError("Product work disappeared during recovery reporting.")
                rules.require_recovery_owner(
                    rules.operation_from_row(row),
                    current_claim_id=row["execution_claim_id"],
                    claim_id=claim_id,
                )
                changed = connection.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET recovery_status = ?
                       WHERE work_id = ? AND status = 'pending' AND execution_claim_id = ?""",
                    (recovery_status, work_id, claim_id),
                ).rowcount
                if changed != 1:
                    raise ProductExecutionClaimLost(
                        "Product execution ownership was lost before recovery reporting."
                    )
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during recovery reporting.")
            return rules.operation_from_row(row)

        return await self._run(record)

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

        def finish(connection: sqlite3.Connection) -> ProductOperation:
            with sqlite_support._transaction(connection):
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
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
                    return operation
                # Terminal rows retain the settling claim so a repeated write can
                # reconstruct this result after acknowledgement loss.
                changed = connection.execute(
                    f"""UPDATE {PRODUCT_OPERATIONS_TABLE}
                       SET status = ?, result = ?, recovery_status = NULL,
                           execution_claim_expires_at = NULL
                       WHERE work_id = ? AND status = 'pending' AND execution_claim_id = ?""",
                    (status, result, work_id, claim_id),
                ).rowcount
                if changed != 1:
                    raise ProductExecutionClaimLost(
                        "Product execution ownership was lost before completion."
                    )
                row = connection.execute(_SELECT_OPERATION, (work_id,)).fetchone()
            if row is None:
                raise RuntimeError("Product work disappeared during completion.")
            return rules.operation_from_row(row)

        return await self._run(finish)


__all__ = ["SQLiteProductOperationStore"]
