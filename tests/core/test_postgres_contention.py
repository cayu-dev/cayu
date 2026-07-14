from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from postgres_contention_support import (
    assert_waiting,
    drop_cayu_tables,
    hold_advisory_xact_lock,
    recorded_revisions,
)

from cayu import (
    Event,
    EventQuery,
    EventType,
    Message,
    PostgresBudgetLedger,
    PostgresSessionStore,
    PostgresTaskStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SessionStatusConflict,
    TaskCreate,
    TaskQuery,
)
from cayu.runtime import BudgetLimit, BudgetReservation, BudgetWindow, ModelPricing, PricingCatalog
from cayu.storage import migrations as schema
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import _SCHEMA_ADVISORY_LOCK_KEY

pytestmark = pytest.mark.usefixtures("postgres_dsn")


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _request(session_id: str) -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id=session_id,
        messages=[Message.text("user", "hi")],
    )


def _expected_revisions() -> list[tuple[int, str, int]]:
    return [(rev.revision, str(rev.kind), rev.compatible_from) for rev in schema.REVISIONS]


def _reservation_budget_limit(max_cost: str, *, key: str | None = None) -> BudgetLimit:
    return BudgetLimit(
        scope="app" if key is None else "agent",
        key=key,
        max_estimated_cost=Decimal(max_cost),
        window=BudgetWindow.all_time(),
        pricing=PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    cache_read_input_per_million=Decimal("0.25"),
                    cache_write_input_per_million=Decimal("1.25"),
                ),
            )
        ),
        reservation=BudgetReservation(
            max_input_tokens=100_000,
            max_output_tokens=50_000,
            max_cache_read_input_tokens=40_000,
            max_cache_write_input_tokens=8_000,
        ),
    )


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


async def _reserve(ledger: PostgresBudgetLedger, limit: BudgetLimit, session_id: str):
    return await ledger.reserve(
        limit=limit,
        session_id=session_id,
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )


async def _append_event_without_commit(
    dsn: str,
    *,
    session_id: str,
    event: Event,
    inserted: asyncio.Event,
    release: asyncio.Event,
) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE cayu_sessions
                SET event_seq = event_seq + 1
                WHERE id = %s
                RETURNING event_seq
                """,
                (session_id,),
            )
            row = await cur.fetchone()
            assert row is not None
            session_order = row[0]
            await cur.execute(
                """
                INSERT INTO cayu_events (
                    session_id,
                    session_order,
                    event_id,
                    event_type,
                    timestamp,
                    agent_name,
                    environment_name,
                    workflow_name,
                    tool_name,
                    payload,
                    event
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    session_order,
                    event.id,
                    str(event.type),
                    event.timestamp,
                    event.agent_name,
                    event.environment_name,
                    event.workflow_name,
                    event.tool_name,
                    json.dumps(event.payload),
                    json.dumps(event.model_dump(mode="json")),
                ),
            )
            inserted.set()
            await release.wait()
        await conn.commit()


async def _expire_task_lease(dsn: str, task_id: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE cayu_tasks
                SET lease_expires_at = now() - interval '1 second'
                WHERE id = %s
                """,
                (task_id,),
            )
        await conn.commit()


def test_postgres_cross_session_event_reader_waits_for_older_open_insert(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await store.create(_request("sess_slow"), identity=_identity())
            await store.create(_request("sess_fast"), identity=_identity())
            inserted = asyncio.Event()
            release = asyncio.Event()
            slow_event = Event(
                id="event_slow",
                type=EventType.SESSION_COMPLETED,
                session_id="sess_slow",
            )
            slow_append = asyncio.create_task(
                _append_event_without_commit(
                    postgres_dsn,
                    session_id="sess_slow",
                    event=slow_event,
                    inserted=inserted,
                    release=release,
                )
            )
            await inserted.wait()

            fast_event = Event(
                id="event_fast",
                type=EventType.SESSION_COMPLETED,
                session_id="sess_fast",
            )
            await store.append_event("sess_fast", fast_event)

            single_session_during_open_insert = await store.query_events(
                EventQuery(
                    session_id="sess_fast",
                    after_sequence=0,
                    event_type=EventType.SESSION_COMPLETED,
                    limit=10,
                )
            )
            during_open_insert = await store.query_events(
                EventQuery(
                    after_sequence=0,
                    event_type=EventType.SESSION_COMPLETED,
                    limit=10,
                )
            )

            release.set()
            await asyncio.wait_for(slow_append, timeout=2.0)
            after_commit = await store.query_events(
                EventQuery(
                    after_sequence=0,
                    event_type=EventType.SESSION_COMPLETED,
                    limit=10,
                )
            )
            return single_session_during_open_insert, during_open_insert, after_commit
        finally:
            await store.close()

    single_session_during_open_insert, during_open_insert, after_commit = asyncio.run(run())

    assert [record.event.id for record in single_session_during_open_insert] == ["event_fast"]
    assert during_open_insert == []
    assert [record.event.id for record in after_commit] == ["event_slow", "event_fast"]
    assert [record.sequence for record in after_commit] == sorted(
        record.sequence for record in after_commit
    )


def test_postgres_batched_initial_event_query_sees_committed_later_insert(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        release: asyncio.Event | None = None
        slow_append: asyncio.Task[None] | None = None
        try:
            await store.create(_request("sess_slow"), identity=_identity())
            await store.create(_request("sess_fast"), identity=_identity())
            inserted = asyncio.Event()
            release = asyncio.Event()
            slow_append = asyncio.create_task(
                _append_event_without_commit(
                    postgres_dsn,
                    session_id="sess_slow",
                    event=Event(
                        id="event_slow",
                        type=EventType.SESSION_COMPLETED,
                        session_id="sess_slow",
                    ),
                    inserted=inserted,
                    release=release,
                )
            )
            await inserted.wait()

            await store.append_event(
                "sess_fast",
                Event(
                    id="event_fast",
                    type=EventType.SESSION_COMPLETED,
                    session_id="sess_fast",
                ),
            )

            dummy_session_ids = tuple(f"missing_{index}" for index in range(500))
            during_open_insert = await store.query_events(
                EventQuery(session_ids=("sess_fast", *dummy_session_ids), limit=10)
            )

            return during_open_insert
        finally:
            if release is not None:
                release.set()
            if slow_append is not None:
                await asyncio.wait_for(slow_append, timeout=2.0)
            await store.close()

    during_open_insert = asyncio.run(run())

    assert [record.event.id for record in during_open_insert] == ["event_fast"]


def test_postgres_session_store_concurrent_transition_status_conflicts(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        first = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        second = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            await first.create(_request("sess_status_race"), identity=_identity())
            results = await asyncio.gather(
                first.transition_status(
                    "sess_status_race",
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                ),
                second.transition_status(
                    "sess_status_race",
                    from_statuses={SessionStatus.PENDING},
                    to_status=SessionStatus.RUNNING,
                ),
                return_exceptions=True,
            )
            loaded = await first.load("sess_status_race")
            return results, loaded
        finally:
            await first.close()
            await second.close()

    results, loaded = asyncio.run(run())

    successes = [result for result in results if not isinstance(result, Exception)]
    conflicts = [result for result in results if isinstance(result, SessionStatusConflict)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert successes[0].status == SessionStatus.RUNNING
    assert loaded is not None
    assert loaded.status == SessionStatus.RUNNING


def test_postgres_budget_ledger_concurrent_reservations_do_not_double_spend(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        setup = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await setup.ensure_schema()
        finally:
            await setup.close()

        first = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        second = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            limit = _reservation_budget_limit("0.25")
            return await asyncio.gather(
                _reserve(first, limit, "sess_budget_a"),
                _reserve(second, limit, "sess_budget_b"),
            )
        finally:
            await first.close()
            await second.close()

    results = asyncio.run(run())

    accepted = [result for result in results if result.accepted]
    rejected = [result for result in results if not result.accepted]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0].record is not None
    assert rejected[0].actual == Decimal("0.44")


def test_postgres_budget_ledger_heartbeat_and_reconcile_serialize_same_reservation(
    postgres_dsn: str,
) -> None:
    async def run():
        import psycopg

        await drop_cayu_tables(postgres_dsn)
        setup = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        limit = _reservation_budget_limit("0.25")
        try:
            reserved = await _reserve(setup, limit, "sess_budget_race")
            assert reserved.record is not None
            reservation_id = reserved.record.reservation_id
        finally:
            await setup.close()

        heartbeat_ledger = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        reconcile_ledger = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            await asyncio.gather(
                heartbeat_ledger.ensure_schema(),
                reconcile_ledger.ensure_schema(),
            )
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as blocker:
                async with blocker.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT reservation_id
                        FROM cayu_budget_reservations
                        WHERE reservation_id = %s
                        FOR UPDATE
                        """,
                        (reservation_id,),
                    )
                    assert await cur.fetchone() == (reservation_id,)

                    heartbeat_task = asyncio.create_task(
                        heartbeat_ledger.heartbeat(reservation_id=reservation_id)
                    )
                    reconcile_task = asyncio.create_task(
                        reconcile_ledger.reconcile(
                            reservation_id=reservation_id,
                            actual_amount=Decimal("0.01"),
                            reason="model completed",
                        )
                    )
                    await assert_waiting(heartbeat_task)
                    await assert_waiting(reconcile_task)

                await blocker.commit()

                heartbeat_result, reconciled = await asyncio.wait_for(
                    asyncio.gather(heartbeat_task, reconcile_task),
                    timeout=5,
                )

            replacement = await _reserve(
                heartbeat_ledger,
                limit,
                "sess_budget_replacement",
            )
            return heartbeat_result, reconciled, replacement
        finally:
            await heartbeat_ledger.close()
            await reconcile_ledger.close()

    heartbeat_result, reconciled, replacement = asyncio.run(run())

    assert type(heartbeat_result) is bool
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")
    assert replacement.accepted is True
    assert replacement.actual == Decimal("0.23")


def test_postgres_budget_ledger_reaps_only_the_advisory_locked_budget(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            clock=clock,
            reservation_ttl_seconds=1,
        )
        try:
            first_limit = _reservation_budget_limit("0.25", key="first")
            second_limit = _reservation_budget_limit("0.25", key="second")
            first = await _reserve(ledger, first_limit, "sess_first_expired")
            second = await _reserve(ledger, second_limit, "sess_second_expired")
            assert first.record is not None
            assert second.record is not None
            clock.value += timedelta(seconds=2)

            replacement = await _reserve(ledger, first_limit, "sess_first_replacement")
            untouched = await ledger.release(
                reservation_id=second.record.reservation_id,
                reason="manual cleanup",
            )
            return replacement, untouched
        finally:
            await ledger.close()

    replacement, untouched = asyncio.run(run())

    assert replacement.accepted is True
    assert untouched.reason == "manual cleanup"


def test_postgres_budget_ledger_concurrent_scoped_reaping_does_not_deadlock(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        setup = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            clock=clock,
            reservation_ttl_seconds=1,
        )
        first_limit = _reservation_budget_limit("0.25")
        second_limit = _reservation_budget_limit("0.25", key="second")
        try:
            assert (await _reserve(setup, first_limit, "sess_first_expired")).accepted
            assert (await _reserve(setup, second_limit, "sess_second_expired")).accepted
        finally:
            await setup.close()

        clock.value += timedelta(seconds=2)
        first = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            clock=clock,
            reservation_ttl_seconds=1,
        )
        second = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            clock=clock,
            reservation_ttl_seconds=1,
        )
        try:
            return await asyncio.wait_for(
                asyncio.gather(
                    _reserve(first, first_limit, "sess_first_replacement"),
                    _reserve(second, second_limit, "sess_second_replacement"),
                ),
                timeout=5,
            )
        finally:
            await first.close()
            await second.close()

    results = asyncio.run(run())

    assert all(result.accepted for result in results)
    assert all(result.actual == Decimal("0.22") for result in results)


def test_postgres_store_schema_reconcile_waits_on_shared_advisory_lock(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        await drop_cayu_tables(postgres_dsn)
        sessions = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        tasks = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            async with hold_advisory_xact_lock(postgres_dsn, _SCHEMA_ADVISORY_LOCK_KEY):
                session_reconcile = asyncio.create_task(sessions.ensure_schema())
                task_reconcile = asyncio.create_task(tasks.ensure_schema())
                await assert_waiting(session_reconcile)
                await assert_waiting(task_reconcile)
            await asyncio.gather(session_reconcile, task_reconcile)
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(run())
    assert asyncio.run(recorded_revisions(postgres_dsn)) == _expected_revisions()


def test_postgres_task_store_races_reclaim_and_claim_one_expired_lease(
    postgres_dsn: str,
) -> None:
    async def run():
        await drop_cayu_tables(postgres_dsn)
        setup = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        first = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        second = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            await setup.create_task(TaskCreate(task_id="task_lease_race", type="review"))
            claimed = await setup.claim_task("stale_worker", lease_seconds=300)
            assert claimed is not None
            await _expire_task_lease(postgres_dsn, "task_lease_race")

            async def worker(store: PostgresTaskStore, worker_id: str):
                reclaimed = await store.reclaim_expired(
                    query=TaskQuery(type="review"),
                    max_reclaims=1,
                )
                claimed = await store.claim_task(
                    worker_id,
                    TaskQuery(type="review"),
                    lease_seconds=300,
                )
                return reclaimed, claimed

            worker_results = await asyncio.gather(
                worker(first, "worker_a"),
                worker(second, "worker_b"),
            )
            loaded = await setup.load_task("task_lease_race")
            return worker_results, loaded
        finally:
            await setup.close()
            await first.close()
            await second.close()

    worker_results, loaded = asyncio.run(run())

    reclaimed_ids = [task.id for reclaimed, _claimed in worker_results for task in reclaimed]
    claimed_tasks = [claimed for _reclaimed, claimed in worker_results if claimed is not None]
    assert reclaimed_ids == ["task_lease_race"]
    assert len(claimed_tasks) == 1
    assert claimed_tasks[0].id == "task_lease_race"
    assert loaded is not None
    assert loaded.worker_id == claimed_tasks[0].worker_id
