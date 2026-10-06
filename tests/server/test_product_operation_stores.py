"""One contract suite for the runtime-owned SQLite and PostgreSQL product stores."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from psycopg.errors import UniqueViolation

from cayu import PostgresProductOperationStore, SQLiteProductOperationStore
from cayu.cli import main
from cayu.server import (
    ProductExecutionClaimLost,
    ProductIdempotencyConflict,
    ProductOperation,
    ProductOperationSettlementConflict,
    ProductOperationStore,
    ProductResultReceipt,
    ProductResultReceiptConflict,
    ServiceIdentityStoreKind,
)
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema

pytestmark = pytest.mark.anyio

LEASE = 60


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture(params=["sqlite", "postgres"])
async def stores(request, tmp_path):
    """Open independent store instances (separate connections) on one database."""

    backend = request.param
    address = (
        str(tmp_path / "product.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    opened = []

    def factory(clock: Clock | None = None):
        if backend == "sqlite":
            store = SQLiteProductOperationStore(address, clock=clock)
        else:
            store = PostgresProductOperationStore(
                address, schema_mode=schema.SchemaMode.CREATE, clock=clock
            )
        opened.append(store)
        return store

    yield factory
    for store in opened:
        await store.close()


def identities(prefix: str = "op") -> dict[str, str]:
    # Postgres databases are shared per module, so every test binds fresh ids.
    token = uuid4().hex
    return {
        "idempotency_key": f"key-{token}",
        "public_id": f"{prefix}_{token}",
        "work_id": f"work_{token}",
        "session_id": f"session_{token}",
        "task_id": f"task_{token}",
    }


async def reserve(store, *, tenant_id: str = "tenant-a", fingerprint: str = "fp", **overrides):
    values = {
        "tenant_id": tenant_id,
        "subject_id": "alice",
        "request_fingerprint": fingerprint,
        "request_text": "work",
        **identities(),
        **overrides,
    }
    return await store.reserve(**values)


def receipt_for(
    operation: ProductOperation,
    *,
    sequence: int = 10,
    result: str = "answer",
    work_id: str | None = None,
) -> ProductResultReceipt:
    return ProductResultReceipt.create(
        work_id=work_id or operation.work_id,
        public_id=operation.public_id,
        request_fingerprint=operation.request_fingerprint,
        session_id=operation.session_id,
        task_id=operation.task_id,
        source_event_id=f"model-completed-{sequence}",
        source_event_sequence=sequence,
        model_step_id="model-step",
        model_attempt_id=f"model-attempt-{sequence}",
        interaction_id="interaction",
        publication_status="completed",
        result=result,
    )


async def claimed(store, clock=None, claim_id: str = "claim-one"):
    reservation = await reserve(store)
    claim = await store.claim_execution(
        work_id=reservation.operation.work_id, claim_id=claim_id, lease_seconds=LEASE
    )
    assert claim is not None and claim.acquired
    return reservation.operation


async def test_stores_implement_the_protocol_and_declare_durability(stores, tmp_path):
    store = stores()
    assert isinstance(store, ProductOperationStore)
    assert store.category is ServiceIdentityStoreKind.DURABLE
    memory = SQLiteProductOperationStore(":memory:")
    try:
        assert memory.category is ServiceIdentityStoreKind.DEVELOPMENT
        reservation = await reserve(memory)
        assert await memory.find(
            tenant_id="tenant-a", public_id=reservation.operation.public_id
        ) == (reservation.operation)
    finally:
        await memory.close()


async def test_reservation_is_idempotent_and_tenant_qualified(stores):
    store = stores()
    ids = identities()
    first = await reserve(store, **ids)
    assert first.created
    assert first.operation.status == "pending" and first.operation.result is None

    replay = await reserve(
        store, **{**ids, **identities(), "idempotency_key": ids["idempotency_key"]}
    )
    assert not replay.created
    assert replay.operation == first.operation

    with pytest.raises(ProductIdempotencyConflict):
        await reserve(store, fingerprint="other", idempotency_key=ids["idempotency_key"])
    with pytest.raises(ProductIdempotencyConflict):
        await reserve(store, tenant_id="tenant-b", idempotency_key=ids["idempotency_key"])

    # Public ids are only resolvable inside their tenant.
    assert await store.find(tenant_id="tenant-a", public_id=ids["public_id"]) == first.operation
    assert await store.find(tenant_id="tenant-b", public_id=ids["public_id"]) is None
    assert await store.find(tenant_id="tenant-a", public_id=ids["session_id"]) is None
    assert await store.find(tenant_id="tenant-a", public_id=f"op_{uuid4().hex}") is None

    # The private continuation lookup uses the exact session index only.
    assert await store.find_by_session_id(session_id=ids["session_id"]) == first.operation
    assert await store.find_by_session_id(session_id=ids["public_id"]) is None
    # Identities no stored row could carry are simply not found.
    for unmatchable in (" ", "op\x00", "o" * 600):
        assert await store.find(tenant_id="tenant-a", public_id=unmatchable) is None
        assert await store.find_by_session_id(session_id=unmatchable) is None


async def test_tenant_isolation_across_operations(stores):
    store = stores()
    a = await reserve(store, tenant_id="tenant-a")
    b = await reserve(store, tenant_id="tenant-b")
    assert await store.find(tenant_id="tenant-a", public_id=a.operation.public_id) == a.operation
    assert await store.find(tenant_id="tenant-b", public_id=b.operation.public_id) == b.operation
    assert await store.find(tenant_id="tenant-a", public_id=b.operation.public_id) is None
    assert await store.find(tenant_id="tenant-b", public_id=a.operation.public_id) is None

    # A reused private identity cannot be rebound to another tenant's work.
    with pytest.raises((sqlite3.IntegrityError, UniqueViolation)):
        await reserve(store, tenant_id="tenant-b", public_id=a.operation.public_id)
    assert await store.find(tenant_id="tenant-b", public_id=a.operation.public_id) is None


async def test_claims_are_exclusive_renewable_and_never_shortened(stores):
    clock = Clock()
    store = stores(clock)
    reservation = await reserve(store)
    work_id = reservation.operation.work_id

    assert (
        await store.claim_execution(work_id=f"work_{uuid4().hex}", claim_id="c", lease_seconds=1)
        is None
    )
    first = await store.claim_execution(work_id=work_id, claim_id="one", lease_seconds=100)
    assert first is not None and first.acquired and first.operation == reservation.operation
    other = await store.claim_execution(work_id=work_id, claim_id="two", lease_seconds=100)
    assert other is not None and not other.acquired
    assert not await store.heartbeat_execution(work_id=work_id, claim_id="two", lease_seconds=100)

    # Renewal and heartbeats with a shorter lease keep the longer expiry.
    renewed = await store.claim_execution(work_id=work_id, claim_id="one", lease_seconds=5)
    assert renewed is not None and renewed.acquired
    assert await store.heartbeat_execution(work_id=work_id, claim_id="one", lease_seconds=5)
    clock.advance(99)
    other = await store.claim_execution(work_id=work_id, claim_id="two", lease_seconds=100)
    assert other is not None and not other.acquired
    assert not await store.heartbeat_execution(
        work_id=f"work_{uuid4().hex}", claim_id="one", lease_seconds=5
    )

    with pytest.raises(ValueError, match="lease_seconds"):
        await store.claim_execution(work_id=work_id, claim_id="one", lease_seconds=0)
    with pytest.raises(ValueError, match="lease_seconds"):
        await store.heartbeat_execution(work_id=work_id, claim_id="one", lease_seconds=True)


async def test_lease_expiry_uses_the_store_clock(stores):
    clock = Clock()
    store = stores(clock)
    operation = await claimed(store, claim_id="abandoned")
    work_id = operation.work_id

    clock.advance(LEASE - 1)
    blocked = await store.claim_execution(work_id=work_id, claim_id="next", lease_seconds=LEASE)
    assert blocked is not None and not blocked.acquired

    # A heartbeat extends ownership from the store's current time.
    assert await store.heartbeat_execution(
        work_id=work_id, claim_id="abandoned", lease_seconds=LEASE
    )
    clock.advance(LEASE - 1)
    blocked = await store.claim_execution(work_id=work_id, claim_id="next", lease_seconds=LEASE)
    assert blocked is not None and not blocked.acquired

    clock.advance(1)
    recovered = await store.claim_execution(work_id=work_id, claim_id="next", lease_seconds=LEASE)
    assert recovered is not None and recovered.acquired

    # The expired owner can no longer renew, publish, report, release, or settle.
    assert not await store.heartbeat_execution(
        work_id=work_id, claim_id="abandoned", lease_seconds=LEASE
    )
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_result_receipt(
            work_id=work_id, claim_id="abandoned", receipt=receipt_for(operation)
        )
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_recovery_status(
            work_id=work_id, claim_id="abandoned", recovery_status="interrupted"
        )
    assert not await store.release_execution(work_id=work_id, claim_id="abandoned")
    with pytest.raises(ProductExecutionClaimLost):
        await store.finish(work_id=work_id, claim_id="abandoned", status="failed", result=None)
    assert await store.heartbeat_execution(work_id=work_id, claim_id="next", lease_seconds=LEASE)


async def test_release_is_idempotent_and_never_clears_a_successor(stores):
    store = stores(Clock())
    operation = await claimed(store, claim_id="one")
    work_id = operation.work_id

    assert await store.release_execution(work_id=work_id, claim_id="one")
    assert await store.release_execution(work_id=work_id, claim_id="one")
    successor = await store.claim_execution(work_id=work_id, claim_id="two", lease_seconds=LEASE)
    assert successor is not None and successor.acquired
    assert not await store.release_execution(work_id=work_id, claim_id="one")
    assert await store.heartbeat_execution(work_id=work_id, claim_id="two", lease_seconds=LEASE)

    await store.finish(work_id=work_id, claim_id="two", status="failed", result=None)
    assert not await store.release_execution(work_id=work_id, claim_id="two")
    with pytest.raises(RuntimeError, match="disappeared"):
        await store.release_execution(work_id=f"work_{uuid4().hex}", claim_id="two")


async def test_result_receipts_are_content_bound_and_claim_fenced(stores):
    clock = Clock()
    store = stores(clock)
    operation = await claimed(store, claim_id="one")
    work_id = operation.work_id
    receipt = receipt_for(operation)

    await store.record_recovery_status(
        work_id=work_id, claim_id="one", recovery_status="runtime_active"
    )
    assert await store.record_result_receipt(work_id=work_id, claim_id="one", receipt=receipt) == (
        receipt
    )
    # Exact repetition reconstructs an ambiguous acknowledgement.
    assert await store.record_result_receipt(work_id=work_id, claim_id="one", receipt=receipt) == (
        receipt
    )
    stored = await store.find(tenant_id="tenant-a", public_id=operation.public_id)
    assert stored is not None
    assert stored.result_receipt == receipt
    assert stored.recovery_status is None
    assert stored.status == "pending" and stored.result is None

    with pytest.raises(ProductResultReceiptConflict):
        await store.record_result_receipt(
            work_id=work_id, claim_id="one", receipt=receipt_for(operation, result="other")
        )
    with pytest.raises(ProductResultReceiptConflict):
        await store.record_result_receipt(
            work_id=work_id,
            claim_id="one",
            receipt=receipt_for(operation, sequence=20, work_id=f"work_{uuid4().hex}"),
        )
    newer = receipt_for(operation, sequence=11, result="newer answer")
    assert await store.record_result_receipt(work_id=work_id, claim_id="one", receipt=newer) == (
        newer
    )
    with pytest.raises(ProductResultReceiptConflict):
        await store.record_result_receipt(work_id=work_id, claim_id="one", receipt=receipt)
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_result_receipt(
            work_id=work_id, claim_id="two", receipt=receipt_for(operation, sequence=12)
        )
    with pytest.raises(TypeError):
        await store.record_result_receipt(
            work_id=work_id, claim_id="one", receipt=newer.model_dump()
        )

    completed = await store.finish(
        work_id=work_id, claim_id="one", status="completed", result="newer answer"
    )
    assert completed.result_receipt == newer
    # The settling claim still reconstructs its receipt; no one may replace it.
    assert await store.record_result_receipt(work_id=work_id, claim_id="one", receipt=newer) == (
        newer
    )
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_result_receipt(
            work_id=work_id, claim_id="one", receipt=receipt_for(operation, sequence=13)
        )


async def test_recovery_status_is_bounded_and_claim_fenced(stores):
    store = stores(Clock())
    operation = await claimed(store, claim_id="one")
    work_id = operation.work_id

    updated = await store.record_recovery_status(
        work_id=work_id, claim_id="one", recovery_status="waiting_for_approval"
    )
    assert updated.recovery_status == "waiting_for_approval"
    assert updated == await store.find(tenant_id="tenant-a", public_id=operation.public_id)
    assert (
        await store.record_recovery_status(
            work_id=work_id, claim_id="one", recovery_status="waiting_for_approval"
        )
        == updated
    )
    with pytest.raises(ValueError, match="recovery_status"):
        await store.record_recovery_status(
            work_id=work_id, claim_id="one", recovery_status="running"
        )
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_recovery_status(
            work_id=work_id, claim_id="two", recovery_status="interrupted"
        )

    failed = await store.finish(work_id=work_id, claim_id="one", status="failed", result=None)
    assert failed.recovery_status is None
    with pytest.raises(ProductExecutionClaimLost):
        await store.record_recovery_status(
            work_id=work_id, claim_id="one", recovery_status="interrupted"
        )


async def test_settlement_is_conditional_on_claim_and_receipt(stores):
    store = stores(Clock())
    operation = await claimed(store, claim_id="one")
    work_id = operation.work_id

    with pytest.raises(ProductOperationSettlementConflict, match="receipt"):
        await store.finish(work_id=work_id, claim_id="one", status="completed", result="answer")
    await store.record_result_receipt(
        work_id=work_id, claim_id="one", receipt=receipt_for(operation)
    )
    with pytest.raises(ProductOperationSettlementConflict, match="receipt"):
        await store.finish(work_id=work_id, claim_id="one", status="completed", result="other")
    with pytest.raises(ProductOperationSettlementConflict, match="public result"):
        await store.finish(work_id=work_id, claim_id="one", status="failed", result="answer")
    with pytest.raises(ProductExecutionClaimLost):
        await store.finish(work_id=work_id, claim_id="two", status="completed", result="answer")
    with pytest.raises(ValueError, match="status"):
        await store.finish(work_id=work_id, claim_id="one", status="pending", result=None)

    completed = await store.finish(
        work_id=work_id, claim_id="one", status="completed", result="answer"
    )
    assert completed.status == "completed" and completed.result == "answer"
    assert completed == await store.find(tenant_id="tenant-a", public_id=operation.public_id)

    # The settling claim reconstructs its write; everything else is refused.
    assert (
        await store.finish(work_id=work_id, claim_id="one", status="completed", result="answer")
        == completed
    )
    with pytest.raises(ProductOperationSettlementConflict, match="different terminal"):
        await store.finish(work_id=work_id, claim_id="one", status="failed", result=None)
    with pytest.raises(ProductExecutionClaimLost):
        await store.finish(work_id=work_id, claim_id="two", status="completed", result="answer")
    assert await store.heartbeat_execution(work_id=work_id, claim_id="one", lease_seconds=LEASE)
    assert not await store.heartbeat_execution(work_id=work_id, claim_id="two", lease_seconds=LEASE)
    terminal = await store.claim_execution(work_id=work_id, claim_id="two", lease_seconds=LEASE)
    assert terminal is not None and not terminal.acquired and terminal.operation == completed

    # Reopening the database observes the same committed authority.
    reopened = stores()
    assert await reopened.find(tenant_id="tenant-a", public_id=operation.public_id) == completed


async def test_concurrent_reservations_of_one_key_create_once(stores):
    instances = [stores() for _ in range(6)]
    key = f"key-{uuid4().hex}"
    reservations = await asyncio.gather(
        *(reserve(store, idempotency_key=key) for store in instances)
    )
    assert sum(reservation.created for reservation in reservations) == 1
    assert len({reservation.operation for reservation in reservations}) == 1


async def test_concurrent_claims_acquire_once_and_settle_once(stores):
    clock = Clock()
    instances = [stores(clock) for _ in range(6)]
    operations = [(await reserve(instances[0])).operation for _ in range(4)]

    for operation in operations:
        claims = await asyncio.gather(
            *(
                store.claim_execution(
                    work_id=operation.work_id, claim_id=f"claim-{index}", lease_seconds=LEASE
                )
                for index, store in enumerate(instances)
            )
        )
        winners = [index for index, claim in enumerate(claims) if claim and claim.acquired]
        assert len(winners) == 1
        owner = f"claim-{winners[0]}"
        await instances[0].record_result_receipt(
            work_id=operation.work_id, claim_id=owner, receipt=receipt_for(operation)
        )

        async def settle(store, claim_id, operation=operation):
            return await store.finish(
                work_id=operation.work_id, claim_id=claim_id, status="completed", result="answer"
            )

        outcomes = await asyncio.gather(
            *(settle(store, f"claim-{index}") for index, store in enumerate(instances)),
            return_exceptions=True,
        )
        settled = [outcome for outcome in outcomes if isinstance(outcome, ProductOperation)]
        assert len(settled) == 1 and settled[0].status == "completed"
        assert all(
            isinstance(outcome, ProductExecutionClaimLost)
            for outcome in outcomes
            if not isinstance(outcome, ProductOperation)
        )


async def test_expired_owner_racing_its_successor_cannot_both_settle(stores):
    clock = Clock()
    first, second = stores(clock), stores(clock)
    for _ in range(4):
        operation = await claimed(first, claim_id="stale")
        await first.record_result_receipt(
            work_id=operation.work_id, claim_id="stale", receipt=receipt_for(operation)
        )
        clock.advance(LEASE)
        successor = await second.claim_execution(
            work_id=operation.work_id, claim_id="fresh", lease_seconds=LEASE
        )
        assert successor is not None and successor.acquired

        outcomes = await asyncio.gather(
            first.finish(
                work_id=operation.work_id, claim_id="stale", status="completed", result="answer"
            ),
            second.finish(
                work_id=operation.work_id, claim_id="fresh", status="failed", result=None
            ),
            first.claim_execution(work_id=operation.work_id, claim_id="stale", lease_seconds=LEASE),
            return_exceptions=True,
        )
        assert isinstance(outcomes[0], ProductExecutionClaimLost)
        assert isinstance(outcomes[1], ProductOperation) and outcomes[1].status == "failed"
        assert outcomes[2] is not None and not outcomes[2].acquired
        stored = await first.find(tenant_id="tenant-a", public_id=operation.public_id)
        assert stored is not None and stored.status == "failed" and stored.result is None


def test_sqlite_store_requires_migration_from_revision_111(tmp_path, monkeypatch):
    path = tmp_path / "revision-111.sqlite"
    revisions = schema.REVISIONS
    monkeypatch.setattr(
        schema, "REVISIONS", tuple(item for item in revisions if item.revision <= 111)
    )
    connection = sqlite_connection.connect(path)
    try:
        sqlite_support.reconcile_schema(
            connection, schema.SchemaMode.MIGRATE, app_min_supported=111
        )
    finally:
        connection.close()
    monkeypatch.setattr(schema, "REVISIONS", revisions)

    with pytest.raises(schema.SchemaTooOld):
        SQLiteProductOperationStore(path)
    assert (
        main(
            [
                "storage",
                "migrate",
                "--sqlite",
                str(path),
                "--waive-backup",
                "--acknowledge-breaking",
                "115",
            ]
        )
        == 0
    )

    async def use_migrated_store() -> bool:
        store = SQLiteProductOperationStore(path, schema_mode=schema.SchemaMode.VALIDATE)
        try:
            return (await reserve(store)).created
        finally:
            await store.close()

    assert asyncio.run(use_migrated_store())


async def test_sqlite_store_refuses_conflicting_product_table(tmp_path):
    path = tmp_path / "conflict.sqlite"
    await SQLiteProductOperationStore(path).close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE cayu_product_operations")
        connection.execute(
            "CREATE TABLE cayu_product_operations (work_id TEXT PRIMARY KEY, status TEXT)"
        )
    with pytest.raises(schema.SchemaError, match="product operation"):
        SQLiteProductOperationStore(path, schema_mode=schema.SchemaMode.VALIDATE)


@pytest.fixture
def fresh_postgres_dsn(postgres_dsn):
    """A private database for tests that need a specific schema history."""

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = f"cayu_product_{uuid4().hex}"
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield make_conninfo(postgres_dsn, dbname=database)
    finally:
        with psycopg.connect(postgres_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


def test_cayu_storage_migrate_upgrades_postgres_to_the_product_revision(
    fresh_postgres_dsn, monkeypatch
):
    from cayu import PostgresSessionStore

    revisions = schema.REVISIONS

    async def create_revision_111() -> None:
        store = PostgresSessionStore(fresh_postgres_dsn, schema_mode=schema.SchemaMode.MIGRATE)
        store._min_required_revision = 111
        try:
            await store.ensure_schema()
        finally:
            await store.close()

    # Seed the historical database without asking today's store to accept it.
    # Restore the real minimum before exercising migration and validation.
    with monkeypatch.context() as seed:
        seed.setattr(schema, "REVISIONS", tuple(item for item in revisions if item.revision <= 111))
        seed.setattr(PostgresSessionStore, "_min_required_revision", 111)
        asyncio.run(create_revision_111())

    async def open_product_store() -> bool:
        store = PostgresProductOperationStore(fresh_postgres_dsn)
        try:
            return (await reserve(store)).created
        finally:
            await store.close()

    with pytest.raises(schema.SchemaTooOld):
        asyncio.run(open_product_store())
    assert (
        main(
            [
                "storage",
                "migrate",
                "--postgres",
                fresh_postgres_dsn,
                "--waive-backup",
                "--acknowledge-breaking",
                "115",
            ]
        )
        == 0
    )
    assert asyncio.run(open_product_store())


def test_postgres_store_refuses_conflicting_product_table(fresh_postgres_dsn):
    import psycopg

    async def scenario() -> None:
        created = PostgresProductOperationStore(
            fresh_postgres_dsn, schema_mode=schema.SchemaMode.CREATE
        )
        try:
            await created.ensure_schema()
        finally:
            await created.close()
        async with await psycopg.AsyncConnection.connect(fresh_postgres_dsn) as connection:
            await connection.execute(
                "ALTER TABLE cayu_product_operations "
                "DROP CONSTRAINT cayu_product_operations_idempotency_key_key"
            )
        reopened = PostgresProductOperationStore(fresh_postgres_dsn)
        try:
            with pytest.raises(schema.SchemaError, match="identity constraints"):
                await reopened.ensure_schema()
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_two_service_processes_settle_each_operation_exactly_once(fresh_postgres_dsn):
    import psycopg

    from cayu.server.service import _product_request_fingerprint

    dsn = fresh_postgres_dsn
    count = 6

    async def prepare() -> list[ProductOperation]:
        store = PostgresProductOperationStore(dsn, schema_mode=schema.SchemaMode.CREATE)
        try:
            await store.ensure_schema()
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                # Independent oracle: every committed pending -> terminal transition.
                await connection.execute(
                    "CREATE TABLE test_product_settlements "
                    "(work_id TEXT NOT NULL, claim_id TEXT NOT NULL, status TEXT NOT NULL)"
                )
                await connection.execute(
                    "CREATE FUNCTION test_record_product_settlement() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN "
                    "INSERT INTO test_product_settlements "
                    "VALUES (NEW.work_id, NEW.execution_claim_id, NEW.status); "
                    "RETURN NEW; END $$"
                )
                await connection.execute(
                    "CREATE TRIGGER test_product_settlement AFTER UPDATE ON "
                    "cayu_product_operations FOR EACH ROW WHEN "
                    "(OLD.status = 'pending' AND NEW.status <> 'pending') "
                    "EXECUTE FUNCTION test_record_product_settlement()"
                )
            operations = []
            for index in range(count):
                request_text = f"request {index}"
                reservation = await reserve(
                    store,
                    request_text=request_text,
                    fingerprint=_product_request_fingerprint(
                        agent_name="assistant", request_text=request_text
                    ),
                )
                operations.append(reservation.operation)
            return operations
        finally:
            await store.close()

    async def scenario() -> tuple[list[ProductOperation], list[dict[str, object]]]:
        # Start both interpreters first; they import while the schema is created.
        workers = [
            await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.server.product_operation_service_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).resolve().parents[2],
            )
            for _ in range(2)
        ]
        try:
            operations = await prepare()
            work_ids = [operation.work_id for operation in operations]
            for worker, order in zip(workers, (work_ids, work_ids[::-1]), strict=True):
                assert worker.stdin is not None
                worker.stdin.write((json.dumps({"dsn": dsn, "work_ids": order}) + "\n").encode())
                await worker.stdin.drain()
            for worker in workers:
                assert worker.stdout is not None and worker.stderr is not None
                if not await asyncio.wait_for(worker.stdout.readline(), 300):
                    pytest.fail((await worker.stderr.read()).decode())
            # Release both service processes together so every operation is contended.
            for worker in workers:
                assert worker.stdin is not None
                worker.stdin.write(b"go\n")
                await worker.stdin.drain()
                worker.stdin.close()
            reports = []
            for worker in workers:
                output, error = await asyncio.wait_for(worker.communicate(), 300)
                assert worker.returncode == 0, error.decode()
                reports.append(json.loads(output))
            return operations, reports
        finally:
            for worker in workers:
                if worker.returncode is None:
                    worker.kill()
                    await worker.wait()

    async def settled(
        operations: list[ProductOperation],
    ) -> tuple[list[tuple[str, str]], list[ProductOperation | None]]:
        async with await psycopg.AsyncConnection.connect(dsn) as connection:
            cursor = await connection.execute(
                "SELECT work_id, status FROM test_product_settlements ORDER BY work_id"
            )
            rows = [(row[0], row[1]) for row in await cursor.fetchall()]
        store = PostgresProductOperationStore(dsn)
        try:
            final = [
                await store.find(tenant_id="tenant-a", public_id=operation.public_id)
                for operation in operations
            ]
        finally:
            await store.close()
        return rows, final

    operations, reports = asyncio.run(scenario())
    rows, final = asyncio.run(settled(operations))

    for report in reports:
        assert set(report["outcomes"].values()) <= {"pending", "completed"}, report
    # Each operation ran provider work once and committed exactly one settlement.
    assert sum(report["provider_calls"] for report in reports) == count
    assert rows == sorted((operation.work_id, "completed") for operation in operations)
    assert all(
        operation is not None
        and operation.status == "completed"
        and operation.result == "settled answer"
        for operation in final
    )
