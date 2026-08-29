from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.core.work_context_store_conformance import (
    assert_work_context_store_conformance,
    checkpoint,
    context,
    recall_delivery,
)

from cayu import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointKey,
    AgentRecallCheckpointMode,
    AgentRecallDeliveryConflict,
    AgentRecallDeliveryState,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextPublicationReceipt,
    InMemoryAgentWorkContextStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    SQLiteAgentWorkContextStore,
    SQLiteKnowledgeStore,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.storage.migrations import SchemaMode


@dataclass(frozen=True)
class _StoreCase:
    name: str
    open: Any
    reset: Any
    reopenable: bool
    clock: _ManualClock


@dataclass
class _ManualClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


async def _drop_postgres_schema(postgres_dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
            )
            for (table,) in await cursor.fetchall():
                await cursor.execute(
                    sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(str(table)))
                )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS cayu_test_block_agent_work_context_head_update() CASCADE"
            )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS "
                "cayu_test_block_agent_recall_checkpoint_head_update() CASCADE"
            )
            await cursor.execute(
                "DROP FUNCTION IF EXISTS "
                "cayu_test_block_agent_recall_delivery_state_insert() CASCADE"
            )
        await connection.commit()


async def _wait_for_postgres_head_lock(
    connection: Any,
    *,
    lock_key: int,
    task: asyncio.Task[Any],
) -> None:
    lock_class_id = (lock_key >> 32) & 0xFFFF_FFFF
    lock_object_id = lock_key & 0xFFFF_FFFF
    for _ in range(1_000):
        if task.done():
            await task
            raise AssertionError(
                "Postgres write completed before reaching its blocked head update."
            )
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_locks
                    WHERE locktype = 'advisory'
                      AND classid::bigint = %s
                      AND objid::bigint = %s
                      AND objsubid = 1
                      AND granted IS FALSE
                )
                """,
                (lock_class_id, lock_object_id),
            )
            row = await cursor.fetchone()
        if row is not None and row[0] is True:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for the Postgres head update to block.")


async def _acquire_postgres_advisory_lock(connection: Any, lock_key: int) -> None:
    await connection.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
    await connection.commit()


async def _release_postgres_advisory_lock(connection: Any, lock_key: int) -> None:
    cursor = await connection.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    row = await cursor.fetchone()
    await connection.commit()
    assert row == (True,)


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def work_context_store_case(request, tmp_path: Path) -> _StoreCase:
    now = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)
    clock = _ManualClock(now)
    if request.param == "memory":

        async def open_memory():
            return InMemoryAgentWorkContextStore(clock=clock)

        async def reset_memory() -> None:
            return None

        return _StoreCase("memory", open_memory, reset_memory, False, clock)
    if request.param == "sqlite":
        location = tmp_path / "work-context.sqlite"

        async def open_sqlite():
            return SQLiteAgentWorkContextStore(location, clock=clock)

        async def reset_sqlite() -> None:
            for path in (location, Path(f"{location}-shm"), Path(f"{location}-wal")):
                path.unlink(missing_ok=True)

        return _StoreCase("sqlite", open_sqlite, reset_sqlite, True, clock)

    postgres_dsn = request.getfixturevalue("postgres_dsn")

    async def open_postgres():
        from cayu import PostgresAgentWorkContextStore

        return PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            clock=clock,
        )

    async def reset_postgres() -> None:
        await _drop_postgres_schema(postgres_dsn)

    return _StoreCase("postgres", open_postgres, reset_postgres, True, clock)


async def _close(store) -> None:
    await store.close()


def test_agent_work_context_store_shared_conformance(work_context_store_case) -> None:
    async def run() -> None:
        await work_context_store_case.reset()
        store = await work_context_store_case.open()
        try:
            await assert_work_context_store_conformance(
                store,
                advance_clock=work_context_store_case.clock.advance,
            )
        finally:
            await _close(store)
        if work_context_store_case.reopenable:
            reopened = await work_context_store_case.open()
            try:
                current = await reopened.load_work_context("task-memory-v51")
                assert current is not None
                assert current.revision == 4
                persisted_checkpoint = await reopened.load_recall_checkpoint(
                    AgentRecallCheckpointKey(
                        agent_id="agent:primary",
                        task_id="task-memory-v51",
                        knowledge_namespace="project:cayu",
                        access_policy_sha256="a" * 64,
                    )
                )
                assert persisted_checkpoint is not None
                assert persisted_checkpoint.revision == 5
                persisted_delivery = await reopened.load_recall_delivery("delivery:3")
                assert persisted_delivery is not None
                assert persisted_delivery.state is AgentRecallDeliveryState.ACKNOWLEDGED
                assert persisted_delivery.delivery.materialized_result().operation_id == (
                    "delivery:process:3"
                )
                claimed_delivery = await reopened.load_recall_delivery("delivery:4")
                released_delivery = await reopened.load_recall_delivery("delivery:5")
                pending_delivery = await reopened.load_recall_delivery("delivery:6")
                assert pending_delivery is not None
                assert pending_delivery.state is AgentRecallDeliveryState.PENDING
                assert pending_delivery.claim is None
                assert claimed_delivery is not None
                assert claimed_delivery.state is AgentRecallDeliveryState.CLAIMED
                assert claimed_delivery.claim is not None
                assert released_delivery is not None
                assert released_delivery.state is AgentRecallDeliveryState.PENDING
                assert released_delivery.release is not None
                assert released_delivery.release.reason == "durable retry evidence"
            finally:
                await _close(reopened)
        await work_context_store_case.reset()

    asyncio.run(run())


def test_agent_work_context_durable_multi_instance_cas(work_context_store_case) -> None:
    if not work_context_store_case.reopenable:
        pytest.skip("The in-memory store has no shared external durability boundary.")

    async def run() -> None:
        await work_context_store_case.reset()
        first_store = await work_context_store_case.open()
        second_store = await work_context_store_case.open()
        try:
            initial = context(revision=1, operation_id="multi-instance:context:create")
            await first_store.publish_work_context(initial, expected_revision=None)
            candidates = (
                context(
                    revision=2,
                    operation_id="multi-instance:context:a",
                    goal="Multi-instance writer A",
                ),
                context(
                    revision=2,
                    operation_id="multi-instance:context:b",
                    goal="Multi-instance writer B",
                ),
            )
            outcomes = await asyncio.gather(
                first_store.publish_work_context(candidates[0], expected_revision=1),
                second_store.publish_work_context(candidates[1], expected_revision=1),
                return_exceptions=True,
            )
            successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            assert len(successes) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], AgentWorkContextConflict)
            current = await first_store.load_work_context(initial.task_id)
            assert current is not None
            assert current in candidates
            assert await second_store.load_work_context(initial.task_id) == current

            initial_checkpoint = checkpoint(
                current,
                revision=1,
                operation_id="multi-instance:checkpoint:create",
            )
            await first_store.advance_recall_checkpoint(
                initial_checkpoint,
                expected_revision=None,
            )
            checkpoint_candidates = (
                checkpoint(
                    current,
                    revision=2,
                    operation_id="multi-instance:checkpoint:a",
                    knowledge_sequence=11,
                    index_readiness_sequence=8,
                    processing_mode=AgentRecallCheckpointMode.DELTA,
                ),
                checkpoint(
                    current,
                    revision=2,
                    operation_id="multi-instance:checkpoint:b",
                    knowledge_sequence=12,
                    index_readiness_sequence=9,
                    processing_mode=AgentRecallCheckpointMode.DELTA,
                ),
            )
            checkpoint_outcomes = await asyncio.gather(
                first_store.advance_recall_checkpoint(
                    checkpoint_candidates[0],
                    expected_revision=1,
                ),
                second_store.advance_recall_checkpoint(
                    checkpoint_candidates[1],
                    expected_revision=1,
                ),
                return_exceptions=True,
            )
            checkpoint_successes = [
                outcome for outcome in checkpoint_outcomes if not isinstance(outcome, BaseException)
            ]
            checkpoint_failures = [
                outcome for outcome in checkpoint_outcomes if isinstance(outcome, BaseException)
            ]
            assert len(checkpoint_successes) == 1
            assert len(checkpoint_failures) == 1
            assert isinstance(checkpoint_failures[0], AgentWorkContextConflict)
            stored_checkpoint = await first_store.load_recall_checkpoint(initial_checkpoint.key())
            assert stored_checkpoint in checkpoint_candidates
            assert (
                await second_store.load_recall_checkpoint(initial_checkpoint.key())
                == stored_checkpoint
            )

            delivery_candidates = (
                await recall_delivery(
                    current,
                    delivery_id="multi-instance:delivery:a",
                    operation_id="multi-instance:delivery:process:a",
                    entry_ids=("multi-instance-entry",),
                ),
                await recall_delivery(
                    current,
                    delivery_id="multi-instance:delivery:b",
                    operation_id="multi-instance:delivery:process:b",
                    entry_ids=("multi-instance-entry",),
                ),
            )
            delivery_outcomes = await asyncio.gather(
                first_store.stage_recall_delivery(delivery_candidates[0]),
                second_store.stage_recall_delivery(delivery_candidates[1]),
                return_exceptions=True,
            )
            delivery_successes = [
                outcome for outcome in delivery_outcomes if not isinstance(outcome, BaseException)
            ]
            delivery_failures = [
                outcome for outcome in delivery_outcomes if isinstance(outcome, BaseException)
            ]
            assert len(delivery_successes) == 1
            assert len(delivery_failures) == 1
            assert isinstance(delivery_failures[0], AgentRecallDeliveryConflict)
            staged_delivery = delivery_successes[0]
            assert staged_delivery.delivery in delivery_candidates
            claim_outcomes = await asyncio.gather(
                first_store.claim_recall_delivery(
                    staged_delivery.delivery.key(),
                    claim_id="multi-instance:claim:a",
                    worker_id="multi-instance:worker:a",
                    lease_seconds=30,
                ),
                second_store.claim_recall_delivery(
                    staged_delivery.delivery.key(),
                    claim_id="multi-instance:claim:b",
                    worker_id="multi-instance:worker:b",
                    lease_seconds=30,
                ),
            )
            claimed = [record for record in claim_outcomes if record is not None]
            assert len(claimed) == 1
            assert claimed[0].state is AgentRecallDeliveryState.CLAIMED
        finally:
            await _close(first_store)
            await _close(second_store)
            await work_context_store_case.reset()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failing_table",
    ("cayu_agent_recall_deliveries", "cayu_agent_recall_delivery_states"),
    ids=("delivery-insert", "state-insert"),
)
def test_sqlite_recall_delivery_stage_rolls_back_every_material_boundary(
    tmp_path: Path,
    failing_table: str,
) -> None:
    async def run() -> None:
        database = tmp_path / "delivery-stage-rollback.sqlite"
        store = SQLiteAgentWorkContextStore(database)
        published = context(revision=1, operation_id="delivery-rollback:sqlite:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-rollback:sqlite",
            operation_id="delivery-rollback:sqlite:process",
            entry_ids=("delivery-rollback-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                f"""
                CREATE TRIGGER cayu_test_fail_agent_recall_delivery_insert
                BEFORE INSERT ON {failing_table}
                BEGIN
                    SELECT RAISE(ABORT, 'test delivery insert failure');
                END
                """
            )
            with pytest.raises(sqlite3.IntegrityError, match="test delivery insert failure"):
                await store.stage_recall_delivery(delivery)
            assert await store.load_recall_checkpoint(delivery.key()) is None
            assert await store.load_recall_delivery(delivery.delivery_id) is None
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "DROP TRIGGER cayu_test_fail_agent_recall_delivery_insert"
            )
            staged = await store.stage_recall_delivery(delivery)
            assert staged.delivery == delivery
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failing_table",
    ("cayu_agent_recall_deliveries", "cayu_agent_recall_delivery_states"),
    ids=("delivery-insert", "state-insert"),
)
def test_postgres_recall_delivery_stage_cancellation_rolls_back_every_material_boundary(
    postgres_dsn: str,
    failing_table: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        lock_key = 7_505_119_600_004
        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=1,
            schema_mode=SchemaMode.CREATE,
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        pending: asyncio.Task[Any] | None = None
        held_lock = False
        published = context(revision=1, operation_id="delivery-rollback:postgres:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-rollback:postgres",
            operation_id="delivery-rollback:postgres:process",
            entry_ids=("delivery-rollback-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_recall_delivery_state_insert()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    f"""
                    CREATE TRIGGER cayu_test_block_agent_recall_delivery_state_insert
                    BEFORE INSERT ON {failing_table}
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_recall_delivery_state_insert()
                    """
                )
            await blocker.commit()
            await _acquire_postgres_advisory_lock(blocker, lock_key)
            held_lock = True
            pending = asyncio.create_task(store.stage_recall_delivery(delivery))
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=lock_key,
                task=pending,
            )
            pending.cancel("cancel staged recall before state publication")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None
            assert await store.load_recall_checkpoint(delivery.key()) is None
            assert await store.load_recall_delivery(delivery.delivery_id) is None
            await _release_postgres_advisory_lock(blocker, lock_key)
            held_lock = False
            staged = await store.stage_recall_delivery(delivery)
            assert staged.delivery == delivery
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            if held_lock:
                await _release_postgres_advisory_lock(blocker, lock_key)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await store.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_sqlite_recall_delivery_rejects_corrupt_denormalized_identity(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = SQLiteAgentWorkContextStore(tmp_path / "delivery-index-corruption.sqlite")
        published = context(revision=1, operation_id="delivery-corruption:sqlite:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-corruption:sqlite",
            operation_id="delivery-corruption:sqlite:process",
            entry_ids=("delivery-corruption-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.stage_recall_delivery(delivery)
            store._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "UPDATE cayu_agent_recall_deliveries "
                "SET processing_result_sha256 = ? WHERE delivery_id = ?",
                ("f" * 64, delivery.delivery_id),
            )
            with pytest.raises(RuntimeError, match="indexes conflict with durable state"):
                await store.load_recall_delivery(delivery.delivery_id)
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_recall_delivery_rejects_corrupt_denormalized_identity(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        published = context(revision=1, operation_id="delivery-corruption:postgres:context")
        delivery = await recall_delivery(
            published,
            delivery_id="delivery-corruption:postgres",
            operation_id="delivery-corruption:postgres:process",
            entry_ids=("delivery-corruption-entry",),
        )
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.stage_recall_delivery(delivery)
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                await connection.execute(
                    "UPDATE cayu_agent_recall_deliveries "
                    "SET processing_result_sha256 = %s WHERE delivery_id = %s",
                    ("f" * 64, delivery.delivery_id),
                )
                await connection.commit()
            with pytest.raises(RuntimeError, match="indexes conflict with durable state"):
                await store.load_recall_delivery(delivery.delivery_id)
        finally:
            await store.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


@pytest.mark.parametrize("use_external_pool", (False, True), ids=("owned-pool", "external-pool"))
def test_postgres_cancellation_rolls_back_context_and_checkpoint_head_updates(
    postgres_dsn: str,
    use_external_pool: bool,
) -> None:
    async def run() -> None:
        import psycopg
        from psycopg_pool import AsyncConnectionPool

        from cayu import PostgresAgentWorkContextStore

        context_lock_key = 7_505_119_600_001
        checkpoint_lock_key = 7_505_119_600_002
        await _drop_postgres_schema(postgres_dsn)
        external_pool = (
            AsyncConnectionPool(
                postgres_dsn,
                min_size=1,
                max_size=1,
                open=False,
            )
            if use_external_pool
            else None
        )
        store = (
            PostgresAgentWorkContextStore(
                pool=external_pool,
                schema_mode=SchemaMode.CREATE,
            )
            if external_pool is not None
            else PostgresAgentWorkContextStore(
                postgres_dsn,
                min_size=1,
                max_size=1,
                schema_mode=SchemaMode.CREATE,
            )
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        held_lock: int | None = None
        pending: asyncio.Task[Any] | None = None
        try:
            initial = context(
                revision=1,
                operation_id="postgres-cancellation:context:create",
            )
            await store.publish_work_context(initial, expected_revision=None)

            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_work_context_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({context_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_work_context_head_update
                    BEFORE UPDATE ON cayu_agent_work_context_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_work_context_head_update()
                    """
                )
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_recall_checkpoint_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({checkpoint_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_recall_checkpoint_head_update
                    BEFORE UPDATE ON cayu_agent_recall_checkpoint_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_recall_checkpoint_head_update()
                    """
                )
            await blocker.commit()

            await _acquire_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = context_lock_key
            successor = context(
                revision=2,
                operation_id="postgres-cancellation:context:append",
                goal="Rollback a cancelled context publication",
            )
            pending = asyncio.create_task(
                store.publish_work_context(successor, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=context_lock_key,
                task=pending,
            )
            pending.cancel("cancel context publication during head update")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None

            assert await store.load_work_context(initial.task_id) == initial
            assert await store.load_work_context(initial.task_id, revision=2) is None
            assert await store.load_work_context_publication(successor.operation_id) is None
            await _release_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = None
            publication = await store.publish_work_context(successor, expected_revision=1)
            assert publication.context == successor

            initial_checkpoint = checkpoint(
                successor,
                revision=1,
                operation_id="postgres-cancellation:checkpoint:create",
            )
            await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)
            await _acquire_postgres_advisory_lock(blocker, checkpoint_lock_key)
            held_lock = checkpoint_lock_key
            successor_checkpoint = checkpoint(
                successor,
                revision=2,
                operation_id="postgres-cancellation:checkpoint:advance",
                knowledge_sequence=11,
                index_readiness_sequence=8,
                processing_mode=AgentRecallCheckpointMode.DELTA,
            )
            pending = asyncio.create_task(
                store.advance_recall_checkpoint(successor_checkpoint, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=checkpoint_lock_key,
                task=pending,
            )
            pending.cancel("cancel checkpoint advancement during head update")
            with pytest.raises(asyncio.CancelledError):
                await pending
            pending = None

            assert (
                await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint
            )
            assert await store.load_recall_checkpoint(initial_checkpoint.key(), revision=2) is None
            await _release_postgres_advisory_lock(blocker, checkpoint_lock_key)
            held_lock = None
            assert (
                await store.advance_recall_checkpoint(
                    successor_checkpoint,
                    expected_revision=1,
                )
                == successor_checkpoint
            )
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
            if held_lock is not None:
                await _release_postgres_advisory_lock(blocker, held_lock)
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await store.close()
            if external_pool is not None:
                await external_pool.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_context_publication_fences_stale_checkpoint(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        context_lock_key = 7_505_119_600_003
        await _drop_postgres_schema(postgres_dsn)
        store = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        held_lock = False
        publication_task: asyncio.Task[Any] | None = None
        checkpoint_task: asyncio.Task[Any] | None = None
        try:
            initial = context(
                revision=1,
                operation_id="postgres-context-fence:context:create",
            )
            await store.publish_work_context(initial, expected_revision=None)
            initial_checkpoint = checkpoint(
                initial,
                revision=1,
                operation_id="postgres-context-fence:checkpoint:create",
            )
            await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)

            async with blocker.cursor() as cursor:
                await cursor.execute(
                    f"""
                    CREATE FUNCTION cayu_test_block_agent_work_context_head_update()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    AS $function$
                    BEGIN
                        PERFORM pg_advisory_xact_lock({context_lock_key});
                        RETURN NEW;
                    END
                    $function$
                    """
                )
                await cursor.execute(
                    """
                    CREATE TRIGGER cayu_test_block_agent_work_context_head_update
                    BEFORE UPDATE ON cayu_agent_work_context_heads
                    FOR EACH ROW
                    EXECUTE FUNCTION cayu_test_block_agent_work_context_head_update()
                    """
                )
                await cursor.execute(
                    "SELECT hashtextextended(%s, 0)",
                    (f"cayu-agent-work-context:task:{initial.task_id}",),
                )
                task_lock_row = await cursor.fetchone()
            await blocker.commit()
            assert task_lock_row is not None
            task_lock_key = int(task_lock_row[0])

            await _acquire_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = True
            successor = context(
                revision=2,
                operation_id="postgres-context-fence:context:append",
                goal="Publish the new current processing basis",
            )
            publication_task = asyncio.create_task(
                store.publish_work_context(successor, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=context_lock_key,
                task=publication_task,
            )

            stale_checkpoint = checkpoint(
                initial,
                revision=2,
                operation_id="postgres-context-fence:checkpoint:stale",
                knowledge_sequence=11,
                index_readiness_sequence=8,
                processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
            )
            checkpoint_task = asyncio.create_task(
                store.advance_recall_checkpoint(stale_checkpoint, expected_revision=1)
            )
            await _wait_for_postgres_head_lock(
                blocker,
                lock_key=task_lock_key,
                task=checkpoint_task,
            )

            await _release_postgres_advisory_lock(blocker, context_lock_key)
            held_lock = False
            publication = await publication_task
            publication_task = None
            assert publication.context == successor
            with pytest.raises(AgentWorkContextConflict, match="stale_work_context_revision"):
                await checkpoint_task
            checkpoint_task = None
            assert (
                await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint
            )
            assert await store.load_recall_checkpoint(initial_checkpoint.key(), revision=2) is None
        finally:
            for pending in (publication_task, checkpoint_task):
                if pending is not None and not pending.done():
                    pending.cancel()
            if held_lock:
                await _release_postgres_advisory_lock(blocker, context_lock_key)
            await asyncio.gather(
                *(
                    pending
                    for pending in (publication_task, checkpoint_task)
                    if pending is not None
                ),
                return_exceptions=True,
            )
            await store.close()
            await blocker.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_work_context_store_rejects_autocommit_pool_and_configuration_drift() -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresAgentWorkContextStore

    autocommit_pool = AsyncConnectionPool(
        "",
        open=False,
        kwargs={"autocommit": True},
    )
    with pytest.raises(TypeError, match="work-context mutations require transactional"):
        PostgresAgentWorkContextStore(pool=autocommit_pool)

    pool = AsyncConnectionPool("", open=False, kwargs={})
    store = PostgresAgentWorkContextStore(pool=pool)
    pool_kwargs = cast("dict[str, Any]", pool.kwargs)
    pool_kwargs["autocommit"] = True

    async def reject_drift() -> None:
        with pytest.raises(TypeError, match="work-context mutations require transactional"):
            await store.load_work_context("task:pool-drift")

    asyncio.run(reject_drift())


def test_agent_work_context_canonicalizes_collections_and_binds_content() -> None:
    value = AgentWorkContext.create(
        task_id="task:canonical",
        goal="Keep deterministic task state",
        revision=1,
        operation_id="operation:canonical",
        published_by="application:test",
        published_at=datetime(2026, 8, 28, tzinfo=UTC),
        scope_ids=("scope:z", "scope:a"),
        entity_ids=("entity:b", "entity:a"),
    )
    assert value.scope_ids == ("scope:a", "scope:z")
    assert value.entity_ids == ("entity:a", "entity:b")
    assert AgentWorkContext.model_validate_json(value.model_dump_json()) == value
    checkpoint_value = checkpoint(
        value,
        revision=1,
        operation_id="checkpoint:canonical",
    )
    assert (
        AgentRecallCheckpoint.model_validate_json(checkpoint_value.model_dump_json())
        == checkpoint_value
    )
    checkpoint_key = checkpoint_value.key()
    assert (
        AgentRecallCheckpointKey.model_validate_json(checkpoint_key.model_dump_json())
        == checkpoint_key
    )
    assert len(checkpoint_key.fingerprint()) == 64
    with pytest.raises(ValidationError, match="content_sha256"):
        value.model_copy(update={"goal": "Altered without a new content identity"})


def test_agent_work_context_publication_receipt_rejects_impossible_no_change() -> None:
    stored = context(revision=1, operation_id="receipt:stored-context")

    with pytest.raises(ValidationError, match="requires an expected revision"):
        AgentWorkContextPublicationReceipt(
            operation_id="receipt:impossible-create-no-change",
            request_sha256="a" * 64,
            expected_revision=None,
            requested_content_sha256=stored.content_sha256,
            changed=False,
            context=stored,
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="requires a distinct operation identity"):
        AgentWorkContextPublicationReceipt(
            operation_id=stored.operation_id,
            request_sha256="a" * 64,
            expected_revision=1,
            requested_content_sha256=stored.content_sha256,
            changed=False,
            context=stored,
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
        )


def test_agent_work_context_rejects_duplicate_and_oversized_values() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        context(
            revision=1,
            operation_id="context:duplicate",
            entity_ids=("same", "same"),
        )
    with pytest.raises(ValidationError, match="goal"):
        AgentWorkContext.create(
            task_id="task:oversized",
            goal="x" * 32_001,
            revision=1,
            operation_id="operation:oversized",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="workflow_iteration"):
        AgentWorkContext.create(
            task_id="task:iteration-overflow",
            goal="Reject unsafe JSON integer overflow",
            revision=1,
            operation_id="operation:iteration-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            workflow_id="workflow:test",
            workflow_phase="bounded",
            workflow_iteration=9_223_372_036_854_775_808,
        )
    with pytest.raises(ValidationError, match="knowledge_sequence"):
        AgentRecallCheckpoint(
            agent_id="agent:overflow",
            task_id="task:overflow",
            knowledge_namespace="project:cayu",
            access_policy_sha256="a" * 64,
            revision=1,
            work_context_revision=1,
            work_context_sha256="b" * 64,
            knowledge_sequence=9_223_372_036_854_775_808,
            index_readiness_sequence=0,
            knowledge_high_water_sequence=9_223_372_036_854_775_808,
            index_readiness_high_water_sequence=0,
            processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
            processing_id="processing:overflow",
            operation_id="operation:overflow",
            updated_by="application:test",
            updated_at=datetime(2026, 8, 28, tzinfo=UTC),
        )

    valid_context = context(revision=1, operation_id="context:frontier-bounds")
    with pytest.raises(ValidationError, match="knowledge_high_water_sequence"):
        checkpoint(
            valid_context,
            revision=1,
            operation_id="checkpoint:future-knowledge",
            knowledge_sequence=11,
            knowledge_high_water_sequence=10,
        )
    with pytest.raises(ValidationError, match="index_readiness_high_water_sequence"):
        checkpoint(
            valid_context,
            revision=1,
            operation_id="checkpoint:future-index",
            index_readiness_sequence=8,
            index_readiness_high_water_sequence=7,
        )
    with pytest.raises(ValidationError, match="task_id"):
        AgentWorkContext.create(
            task_id="x" * 513,
            goal="Reject oversized durable identity",
            revision=1,
            operation_id="operation:identity-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="cannot contain more than 128"):
        AgentWorkContext.create(
            task_id="task:collection-overflow",
            goal="Reject unbounded collections",
            revision=1,
            operation_id="operation:collection-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            entity_ids=tuple(f"entity:{index}" for index in range(129)),
        )
    with pytest.raises(ValidationError, match="serialized byte limit"):
        AgentWorkContext.create(
            task_id="task:record-overflow",
            goal="Reject oversized aggregate content",
            revision=1,
            operation_id="operation:record-overflow",
            published_by="application:test",
            published_at=datetime(2026, 8, 28, tzinfo=UTC),
            entity_ids=tuple(f"entity:{index:03d}:" + "x" * 2_990 for index in range(128)),
        )

    valid = context(revision=1, operation_id="context:strict-input")
    invalid_payload = valid.model_dump(mode="python")
    invalid_payload["revision"] = "1"
    with pytest.raises(ValidationError, match="revision"):
        AgentWorkContext.model_validate(invalid_payload)


def test_sqlite_revision_69_adds_empty_work_context_storage_without_backfill(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-68-to-69-populated.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="revision-67-entry",
                    text="Preserve data without inventing agent work context.",
                )
            )
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        for table in (
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_releases",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_deliveries",
            "cayu_agent_recall_checkpoint_heads",
            "cayu_agent_recall_checkpoints",
            "cayu_agent_work_context_publications",
            "cayu_agent_work_context_heads",
            "cayu_agent_work_context_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision IN (69, 70, 71)")
        connection.execute("PRAGMA user_version = 68")
        connection.commit()
    finally:
        connection.close()

    store = SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)
    asyncio.run(store.close())

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            schema_migrations.LATEST_REVISION,
        )
        assert connection.execute(
            "SELECT text FROM cayu_knowledge_revisions "
            "WHERE entry_id = 'revision-67-entry' AND revision = 1"
        ).fetchone() == ("Preserve data without inventing agent work context.",)
        for table in (
            "cayu_agent_work_context_revisions",
            "cayu_agent_work_context_publications",
            "cayu_agent_recall_checkpoints",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (schema_migrations.LATEST_REVISION,)
    finally:
        connection.close()


def test_sqlite_revision_69_rejects_malformed_work_context_storage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-69-malformed-work-context.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE cayu_agent_work_context_publications")
        connection.execute(
            "CREATE TABLE cayu_agent_work_context_publications (operation_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
        SQLiteAgentWorkContextStore(database)


def test_sqlite_revision_69_primary_identities_are_non_null(tmp_path: Path) -> None:
    database = tmp_path / "revision-69-non-null-primary-identities.sqlite"

    async def seed() -> AgentWorkContext:
        store = SQLiteAgentWorkContextStore(database)
        value = context(revision=1, operation_id="sqlite-non-null:context")
        try:
            await store.publish_work_context(value, expected_revision=None)
        finally:
            await store.close()
        return value

    stored = asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            connection.execute(
                "INSERT INTO cayu_agent_work_context_heads "
                "(task_id, current_revision) VALUES (NULL, 1)"
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            connection.execute(
                "INSERT INTO cayu_agent_work_context_publications ("
                "operation_id, task_id, request_sha256, context_revision, "
                "changed, receipt_json, committed_at"
                ") VALUES (NULL, ?, ?, 1, 0, '{}', ?)",
                (
                    stored.task_id,
                    "a" * 64,
                    "2026-08-28T00:00:00+00:00",
                ),
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "malformation",
    (
        "nocase_identity",
        "split_foreign_key",
        "missing_revision_check",
        "nullable_primary_identity",
    ),
)
def test_sqlite_revision_69_rejects_subtle_work_context_schema_conflicts(
    tmp_path: Path,
    malformation: str,
) -> None:
    database = tmp_path / f"revision-69-{malformation}.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    ddl = sqlite_support._MIGRATION_STEPS[69]
    if malformation == "nocase_identity":
        malformed_ddl = ddl.replace("COLLATE BINARY", "COLLATE NOCASE")
    elif malformation == "split_foreign_key":
        malformed_ddl = ddl.replace(
            """FOREIGN KEY (task_id, current_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT""",
            """FOREIGN KEY (task_id)
                REFERENCES cayu_agent_work_context_revisions(task_id) ON DELETE RESTRICT,
            FOREIGN KEY (current_revision)
                REFERENCES cayu_agent_work_context_revisions(revision) ON DELETE RESTRICT""",
            1,
        )
    elif malformation == "missing_revision_check":
        prefix, marker, checkpoint_ddl = ddl.partition(
            "CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoints"
        )
        assert marker
        checkpoint_ddl = checkpoint_ddl.replace(
            """revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),""",
            "revision INTEGER NOT NULL,",
            1,
        )
        malformed_ddl = prefix + marker + checkpoint_ddl
    else:
        malformed_ddl = ddl.replace(
            "task_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY",
            "task_id TEXT COLLATE BINARY PRIMARY KEY",
            1,
        )
    assert malformed_ddl != ddl

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_checkpoint_heads",
            "cayu_agent_recall_checkpoints",
            "cayu_agent_work_context_publications",
            "cayu_agent_work_context_heads",
            "cayu_agent_work_context_revisions",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.executescript(malformed_ddl)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
        SQLiteAgentWorkContextStore(database)


def test_postgres_revision_69_adds_empty_work_context_storage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore, PostgresKnowledgeStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await creator.ensure_schema()
            await creator.create_entry(
                KnowledgeEntry(
                    id="revision-67-entry",
                    text="Preserve data without inventing agent work context.",
                )
            )
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_states")
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_releases")
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_claims")
                await cursor.execute("DROP TABLE cayu_agent_recall_deliveries")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoint_heads")
                await cursor.execute("DROP TABLE cayu_agent_recall_checkpoints")
                await cursor.execute("DROP TABLE cayu_agent_work_context_publications")
                await cursor.execute("DROP TABLE cayu_agent_work_context_heads")
                await cursor.execute("DROP TABLE cayu_agent_work_context_revisions")
                await cursor.execute(
                    "DELETE FROM cayu_schema_migrations WHERE revision IN (69, 70, 71)"
                )
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions "
                "WHERE entry_id = 'revision-67-entry' AND revision = 1"
            )
            assert await cursor.fetchone() == (
                "Preserve data without inventing agent work context.",
            )
            for table in (
                "cayu_agent_work_context_revisions",
                "cayu_agent_work_context_publications",
                "cayu_agent_recall_checkpoints",
            ):
                await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                assert await cursor.fetchone() == (0,)

        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_69_rejects_malformed_work_context_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_work_context_publications")
                await cursor.execute(
                    "CREATE TABLE cayu_agent_work_context_publications "
                    "(operation_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_69_rejects_missing_checkpoint_revision_constraint(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "ALTER TABLE cayu_agent_recall_checkpoints DROP CONSTRAINT "
                    "cayu_agent_recall_checkpoints_revision_check"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="work-context/checkpoint contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_sqlite_revision_71_adds_empty_delivery_storage_without_backfill(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revision-70-to-71-with-checkpoint.sqlite"
    published = context(revision=1, operation_id="revision-71:sqlite:context")
    processed = checkpoint(
        published,
        revision=1,
        operation_id="revision-71:sqlite:checkpoint",
    )

    async def seed() -> None:
        store = SQLiteAgentWorkContextStore(database)
        try:
            await store.publish_work_context(published, expected_revision=None)
            await store.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_releases",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_deliveries",
        ):
            connection.execute(f"DROP TABLE {table}")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 71")
        connection.execute("PRAGMA user_version = 70")
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteAgentWorkContextStore(database, schema_mode=SchemaMode.MIGRATE)

    async def verify() -> None:
        try:
            assert await migrated.load_work_context(published.task_id) == published
            assert await migrated.load_recall_checkpoint(processed.key()) == processed
        finally:
            await migrated.close()

    asyncio.run(verify())
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (71,)
        for table in (
            "cayu_agent_recall_deliveries",
            "cayu_agent_recall_delivery_states",
            "cayu_agent_recall_delivery_claims",
            "cayu_agent_recall_delivery_releases",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
    finally:
        connection.close()


def test_sqlite_revision_71_rejects_malformed_delivery_storage(tmp_path: Path) -> None:
    database = tmp_path / "revision-71-malformed-delivery.sqlite"
    store = SQLiteAgentWorkContextStore(database)
    asyncio.run(store.close())
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE cayu_agent_recall_delivery_states")
        connection.execute(
            "CREATE TABLE cayu_agent_recall_delivery_states (delivery_id TEXT PRIMARY KEY)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="recall-delivery contract"):
        SQLiteAgentWorkContextStore(database)


def test_postgres_revision_71_adds_empty_delivery_storage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        published = context(revision=1, operation_id="revision-71:postgres:context")
        processed = checkpoint(
            published,
            revision=1,
            operation_id="revision-71:postgres:checkpoint",
        )
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.publish_work_context(published, expected_revision=None)
            await creator.advance_recall_checkpoint(processed, expected_revision=None)
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                for table in (
                    "cayu_agent_recall_delivery_states",
                    "cayu_agent_recall_delivery_releases",
                    "cayu_agent_recall_delivery_claims",
                    "cayu_agent_recall_deliveries",
                ):
                    await cursor.execute(f"DROP TABLE {table}")
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision = 71")
            await connection.commit()

        migrator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            assert await migrator.load_work_context(published.task_id) == published
            assert await migrator.load_recall_checkpoint(processed.key()) == processed
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (71,)
            for table in (
                "cayu_agent_recall_deliveries",
                "cayu_agent_recall_delivery_states",
                "cayu_agent_recall_delivery_claims",
                "cayu_agent_recall_delivery_releases",
            ):
                await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                assert await cursor.fetchone() == (0,)
        await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_71_rejects_malformed_delivery_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresAgentWorkContextStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_agent_recall_delivery_states")
                await cursor.execute(
                    "CREATE TABLE cayu_agent_recall_delivery_states (delivery_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresAgentWorkContextStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match="recall-delivery contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(run())
