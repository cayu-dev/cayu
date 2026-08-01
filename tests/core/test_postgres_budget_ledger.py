"""Postgres budget-ledger parity tests.

Mirror the SQLite ledger assertions in ``test_usage.py`` against a real
Dockerized Postgres. They skip automatically when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.core._budget_ledger_contract import (
    assert_crash_safe_dispatch_and_settlement_outbox,
    assert_idempotent_terminal_settlements,
    assert_portable_text_boundaries,
    assert_prepriced_reservation_stores_only_durable_billing_identity,
    assert_reservation_identity_collision_is_rejected,
    assert_runtime_publishes_cross_session_ttl_release,
    assert_runtime_reconstructs_dispatch_fence_acknowledgement,
)
from tests.core._execution_unit_fixtures import model_attempt_identity

from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    ModelPrice,
    PriceBook,
)
from cayu.runtime.budgets import budget_limits_for_session
from cayu.runtime.sessions import BudgetReservationIdentityConflict

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_budget_settlements",
    "cayu_budget_reservations",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _reservation_budget_limit(
    max_cost: str = "1",
    *,
    window: BudgetWindow | str | None = None,
    key: str | None = None,
) -> BudgetLimit:
    return BudgetLimit(
        scope="app" if key is None else "agent",
        key=key,
        max_estimated_cost=Decimal(max_cost),
        window=BudgetWindow.all_time() if window is None else window,
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
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


async def _drop_all(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_ledger(dsn: str, **kwargs):
    from cayu import PostgresBudgetLedger
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresBudgetLedger(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
        **kwargs,
    )


def _run(dsn: str, coro_factory, **ledger_kwargs) -> object:
    async def runner():
        await _drop_all(dsn)
        ledger = _new_ledger(dsn, **ledger_kwargs)
        try:
            return await coro_factory(ledger)
        finally:
            await ledger.close()

    return asyncio.run(runner())


async def _reserve(ledger, limit, session_id: str):
    return await ledger.reserve(
        limit=limit,
        session_id=session_id,
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        model_attempt_identity=model_attempt_identity(),
    )


def test_postgres_budget_ledger_reserves_reconciles_and_releases(postgres_dsn) -> None:
    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        first = await _reserve(ledger, limit, "sess_1")
        assert first.accepted is True
        assert first.record is not None
        blocked = await _reserve(ledger, limit, "sess_2")
        assert blocked.accepted is False
        reconciled = await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
        )
        retry = await _reserve(ledger, limit, "sess_2")
        assert retry.accepted is True
        assert retry.record is not None
        released = await ledger.release(
            reservation_id=retry.record.reservation_id,
            reason="unused",
        )
        return blocked, reconciled, released

    blocked, reconciled, released = _run(postgres_dsn, ops)

    assert blocked.actual == Decimal("0.44")
    assert reconciled.released_amount == Decimal("0.21")
    assert released.status == "released"


def test_postgres_budget_ledger_terminal_settlements_are_idempotent(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))

    async def ops(ledger):
        await assert_idempotent_terminal_settlements(
            ledger,
            _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.rolling(seconds=60),
            ),
            clock=clock,
        )

    _run(postgres_dsn, ops, clock=clock)


def test_postgres_budget_ledger_has_crash_safe_settlement_outbox(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))

    async def ops(ledger):
        await assert_crash_safe_dispatch_and_settlement_outbox(
            ledger,
            _reservation_budget_limit(max_cost="0.25"),
            clock=clock,
            ttl_seconds=60,
        )

    _run(
        postgres_dsn,
        ops,
        clock=clock,
        reservation_ttl_seconds=60,
    )


def test_postgres_budget_ledger_separates_pricing_and_durable_billing_identity(
    postgres_dsn,
) -> None:
    async def ops(ledger):
        await assert_prepriced_reservation_stores_only_durable_billing_identity(ledger)

    _run(postgres_dsn, ops)


def test_postgres_budget_ledger_reconstructs_dispatch_fence_acknowledgement(
    postgres_dsn,
) -> None:
    async def ops(ledger):
        await assert_runtime_reconstructs_dispatch_fence_acknowledgement(
            ledger,
            _reservation_budget_limit(max_cost="0.25"),
        )

    _run(postgres_dsn, ops)


def test_postgres_budget_ledger_publishes_cross_session_ttl_release(
    postgres_dsn,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))

    async def ops(ledger):
        await assert_runtime_publishes_cross_session_ttl_release(
            ledger,
            _reservation_budget_limit(max_cost="0.25"),
            clock=clock,
            ttl_seconds=60,
        )

    _run(
        postgres_dsn,
        ops,
        clock=clock,
        reservation_ttl_seconds=60,
    )


def test_postgres_revision_twenty_five_refuses_ambiguous_active_reservations(
    postgres_dsn,
) -> None:
    async def verify_rejection() -> None:
        import psycopg

        from cayu import PostgresBudgetLedger
        from cayu.storage.migrations import SchemaMode

        await _drop_all(postgres_dsn)
        creator = PostgresBudgetLedger(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            result = await creator.reserve(
                limit=_reservation_budget_limit(),
                session_id="sess_active_revision_25",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                model_attempt_identity=model_attempt_identity(),
            )
            assert result.accepted is True
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP TABLE cayu_budget_settlements")
                await cur.execute(
                    "ALTER TABLE cayu_budget_reservations "
                    "DROP COLUMN environment_name, "
                    "DROP COLUMN settlement_event_payload, "
                    "DROP COLUMN settlement_fallback, "
                    "DROP COLUMN dispatch_id, "
                    "DROP COLUMN dispatched_at"
                )
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 25")
            await conn.commit()

        migrator = PostgresBudgetLedger(
            postgres_dsn,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(
                psycopg.errors.RaiseException,
                match="cannot migrate active budget reservations",
            ):
                await migrator.load_settlement("settlement-probe")
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT revision FROM cayu_schema_migrations WHERE revision = 25")
            assert await cur.fetchone() is None
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_budget_reservations' "
                "AND column_name IN ('dispatch_id', 'dispatched_at')"
            )
            assert await cur.fetchall() == []

    async def scenario() -> None:
        try:
            await verify_rejection()
        finally:
            await _drop_all(postgres_dsn)

    asyncio.run(scenario())


def test_postgres_budget_ledger_rejects_reservation_identity_collision(postgres_dsn) -> None:
    async def ops(ledger):
        await assert_reservation_identity_collision_is_rejected(
            ledger,
            _reservation_budget_limit(max_cost="1"),
        )

    _run(postgres_dsn, ops)


def test_postgres_budget_ledger_rejects_nonportable_text(postgres_dsn) -> None:
    async def ops(ledger):
        await assert_portable_text_boundaries(
            ledger,
            _reservation_budget_limit(max_cost="0.25"),
        )

    _run(postgres_dsn, ops)


def test_postgres_budget_ledger_persists_reservation_identity_claims_across_instances(
    postgres_dsn,
) -> None:
    async def runner() -> None:
        await _drop_all(postgres_dsn)
        first = _new_ledger(postgres_dsn)
        second = _new_ledger(postgres_dsn)
        claim = {
            "reservation_id": "bres_shared_identity",
            "publication_session_id": "sess_owner",
            "publication_id": "event_owner",
        }
        try:
            await first.claim_reservation_identity(**claim)
            await second.claim_reservation_identity(**claim)
            with pytest.raises(
                BudgetReservationIdentityConflict,
                match="reused a reservation identity",
            ):
                await second.claim_reservation_identity(
                    reservation_id=claim["reservation_id"],
                    publication_session_id="sess_colliding",
                    publication_id="event_colliding",
                )
        finally:
            await first.close()
            await second.close()

    asyncio.run(runner())


def test_postgres_budget_ledger_survives_ledger_restart(postgres_dsn) -> None:
    # The whole point of the durable ledger: reservations written by one worker
    # bind budget for a different worker process on a fresh connection pool.
    async def runner():
        await _drop_all(postgres_dsn)
        first_worker = _new_ledger(postgres_dsn)
        limit = _reservation_budget_limit(max_cost="0.25")
        try:
            first = await _reserve(first_worker, limit, "sess_1")
            assert first.accepted is True
        finally:
            await first_worker.close()
        second_worker = _new_ledger(postgres_dsn)
        try:
            return await _reserve(second_worker, limit, "sess_2")
        finally:
            await second_worker.close()

    blocked = asyncio.run(runner())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")


def test_postgres_budget_ledger_reconstructs_and_partitions_exact_limit_ids(
    postgres_dsn,
) -> None:
    async def runner():
        await _drop_all(postgres_dsn)
        configured = _reservation_budget_limit(max_cost="0.25")
        policy = BudgetPolicy(limits=(configured, configured))
        first_resolution = budget_limits_for_session(
            policy=policy,
            agent_name="assistant",
            causal_budget_id="job_1",
        )
        first_worker = _new_ledger(postgres_dsn)
        try:
            first = await _reserve(first_worker, first_resolution[0], "sess_1")
            parallel = await _reserve(first_worker, first_resolution[1], "sess_1")
        finally:
            await first_worker.close()

        reconstructed = budget_limits_for_session(
            policy=BudgetPolicy.model_validate(policy.model_dump(mode="json")),
            agent_name="assistant",
            causal_budget_id="job_1",
        )
        changed = budget_limits_for_session(
            policy=BudgetPolicy(limits=(_reservation_budget_limit(max_cost="0.30"),)),
            agent_name="assistant",
            causal_budget_id="job_1",
        )[0]
        second_worker = _new_ledger(postgres_dsn)
        try:
            repeated = await _reserve(second_worker, reconstructed[0], "sess_2")
            changed_result = await _reserve(second_worker, changed, "sess_3")
        finally:
            await second_worker.close()
        return first_resolution, reconstructed, first, parallel, repeated, changed_result

    first_resolution, reconstructed, first, parallel, repeated, changed_result = asyncio.run(
        runner()
    )

    assert first_resolution[0].budget_limit_id != first_resolution[1].budget_limit_id
    assert [limit.budget_limit_id for limit in reconstructed] == [
        limit.budget_limit_id for limit in first_resolution
    ]
    assert first.accepted is True
    assert parallel.accepted is True
    assert repeated.accepted is False
    assert changed_result.accepted is True


def test_postgres_budget_ledger_does_not_infer_identity_for_existing_rows(
    postgres_dsn,
) -> None:
    async def runner() -> None:
        import psycopg

        from cayu import PostgresBudgetLedger
        from cayu.storage.migrations import SchemaMode

        await _drop_all(postgres_dsn)
        limit = _reservation_budget_limit(max_cost="0.25")
        creator = _new_ledger(postgres_dsn)
        try:
            existing = await _reserve(creator, limit, "sess_legacy")
            assert existing.record is not None
            reservation_id = existing.record.reservation_id
            await creator.reconcile(
                reservation_id=reservation_id,
                actual_amount=Decimal("0.01"),
                reason="terminal before schema migration",
            )
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP INDEX idx_cayu_budget_reservations_limit")
                await cur.execute(
                    "ALTER TABLE cayu_budget_reservations DROP COLUMN budget_limit_id"
                )
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 22")
            await conn.commit()

        migrated = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(RuntimeError, match="pre-identity reservations"):
                await _reserve(migrated, limit, "sess_new")
            with pytest.raises(
                RuntimeError,
                match="predates durable budget-limit identity",
            ):
                await migrated.reconcile(
                    reservation_id=reservation_id,
                    actual_amount=Decimal("0.01"),
                )
        finally:
            await migrated.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT budget_limit_id, status FROM cayu_budget_reservations "
                "WHERE reservation_id = %s",
                (reservation_id,),
            )
            assert await cur.fetchone() == (None, "reconciled")

    asyncio.run(runner())


def test_postgres_budget_ledger_fails_closed_on_missing_attempt_identity(
    postgres_dsn,
) -> None:
    async def runner() -> None:
        import psycopg

        from cayu import PostgresBudgetLedger
        from cayu.storage.migrations import SchemaMode

        await _drop_all(postgres_dsn)
        limit = _reservation_budget_limit(max_cost="0.25")
        creator = _new_ledger(postgres_dsn)
        try:
            existing = await _reserve(creator, limit, "sess_legacy_attempt")
            assert existing.record is not None
            reservation_id = existing.record.reservation_id
        finally:
            await creator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_budget_reservations "
                    "SET model_step_id = NULL, model_attempt_id = NULL "
                    "WHERE reservation_id = %s",
                    (reservation_id,),
                )
            await conn.commit()

        migrated = PostgresBudgetLedger(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(RuntimeError, match="predates durable model-attempt identity"):
                await migrated.reconcile(
                    reservation_id=reservation_id,
                    actual_amount=Decimal("0.01"),
                )
        finally:
            await migrated.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT model_step_id, model_attempt_id, status "
                "FROM cayu_budget_reservations WHERE reservation_id = %s",
                (reservation_id,),
            )
            assert await cur.fetchone() == (None, None, "active")

    asyncio.run(runner())


def test_postgres_budget_ledger_window_bounds_active_reservations(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        rolling_limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.rolling(seconds=60),
        )
        all_time_limit = _reservation_budget_limit(max_cost="0.25")
        rolling_first = await _reserve(ledger, rolling_limit, "sess_rolling_1")
        all_time_first = await _reserve(ledger, all_time_limit, "sess_all_time_1")
        clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        blocked_now = await _reserve(ledger, rolling_limit, "sess_rolling_blocked")
        clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
        rolling_second = await _reserve(ledger, rolling_limit, "sess_rolling_2")
        all_time_second = await _reserve(ledger, all_time_limit, "sess_all_time_2")
        return rolling_first, blocked_now, rolling_second, all_time_first, all_time_second

    rolling_first, blocked_now, rolling_second, all_time_first, all_time_second = _run(
        postgres_dsn,
        ops,
        clock=clock,
    )

    assert rolling_first.accepted is True
    assert blocked_now.accepted is False
    assert blocked_now.actual == Decimal("0.44")
    assert rolling_second.accepted is False
    assert rolling_second.actual == Decimal("0.44")
    assert all_time_first.accepted is True
    assert all_time_second.accepted is False
    assert all_time_second.actual == Decimal("0.44")


def test_postgres_budget_ledger_keeps_active_reservation_across_calendar_boundary(
    postgres_dsn,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 23, 59, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.calendar(period="day", timezone="UTC"),
        )
        first = await _reserve(ledger, limit, "sess_before_midnight")
        clock.value = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
        blocked = await _reserve(ledger, limit, "sess_after_midnight")
        return first, blocked

    first, blocked = _run(postgres_dsn, ops, clock=clock)

    assert first.accepted is True
    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")


def test_postgres_budget_ledger_uses_reconciliation_time_for_rolling_window(
    postgres_dsn,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.rolling(seconds=60),
        )
        first = await _reserve(ledger, limit, "sess_1")
        assert first.record is not None
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.22"),
            occurred_at=datetime(2026, 1, 1, 12, 2, tzinfo=UTC),
        )
        clock.value = datetime(2026, 1, 1, 12, 2, 30, tzinfo=UTC)
        blocked = await _reserve(ledger, limit, "sess_2")
        clock.value = datetime(2026, 1, 1, 12, 3, 1, tzinfo=UTC)
        accepted = await _reserve(ledger, limit, "sess_3")
        return blocked, accepted

    blocked, accepted = _run(postgres_dsn, ops, clock=clock)

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")
    assert accepted.accepted is True
    assert accepted.actual == Decimal("0.22")


def test_postgres_budget_ledger_uses_reconciliation_time_for_calendar_window(
    postgres_dsn,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.calendar(period="day", timezone="UTC"),
        )
        first = await _reserve(ledger, limit, "sess_1")
        assert first.record is not None
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.22"),
            occurred_at=datetime(2026, 1, 1, 23, 59, tzinfo=UTC),
        )
        clock.value = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
        next_day = await _reserve(ledger, limit, "sess_2")
        active = await _reserve(ledger, limit, "sess_3")
        return next_day, active

    next_day, active = _run(postgres_dsn, ops, clock=clock)

    assert next_day.accepted is True
    assert next_day.window.storage_key == "calendar:day:UTC"
    assert active.accepted is False
    assert active.actual == Decimal("0.44")


def test_postgres_budget_ledger_does_not_reap_dispatched_reservations(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await _reserve(ledger, limit, "sess_orphaned")
        assert orphaned.accepted is True
        assert orphaned.record is not None
        await ledger.mark_dispatched(
            reservation_ids=(orphaned.record.reservation_id,),
            dispatch_id="dispatch:postgres:expired",
        )
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        blocked = await _reserve(ledger, limit, "sess_blocked")
        reconciled = await ledger.reconcile(
            reservation_id=orphaned.record.reservation_id,
            actual_amount=Decimal("0.01"),
        )
        recovered = await _reserve(ledger, limit, "sess_recovered")
        return blocked, reconciled, recovered

    blocked, reconciled, recovered = _run(
        postgres_dsn,
        ops,
        clock=clock,
        reservation_ttl_seconds=60,
    )

    assert blocked.accepted is False
    assert recovered.accepted is True
    assert recovered.actual == Decimal("0.23")
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")


def test_postgres_budget_ledger_heartbeat_keeps_live_reservation_active(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        first = await _reserve(ledger, limit, "sess_live")
        assert first.record is not None
        clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        assert await ledger.heartbeat(reservation_id=first.record.reservation_id) is True
        clock.value = datetime(2026, 1, 1, 12, 1, 1, tzinfo=UTC)
        blocked = await _reserve(ledger, limit, "sess_blocked")
        clock.value = datetime(2026, 1, 1, 12, 1, 30, tzinfo=UTC)
        late_heartbeat = await ledger.heartbeat(reservation_id=first.record.reservation_id)
        recovered = await _reserve(ledger, limit, "sess_recovered")
        return blocked, late_heartbeat, recovered

    blocked, late_heartbeat, recovered = _run(
        postgres_dsn,
        ops,
        clock=clock,
        reservation_ttl_seconds=60,
    )

    assert blocked.accepted is False
    assert late_heartbeat is False
    assert recovered.accepted is True


def test_postgres_budget_ledger_release_tolerates_ttl_reaped_reservation(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await _reserve(ledger, limit, "sess_orphaned")
        assert orphaned.accepted is True
        assert orphaned.record is not None
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        recovered = await _reserve(ledger, limit, "sess_recovered")
        released = await ledger.release(
            reservation_id=orphaned.record.reservation_id,
            reason="cleanup",
        )
        return recovered, released

    recovered, released = _run(postgres_dsn, ops, clock=clock, reservation_ttl_seconds=60)

    assert recovered.accepted is True
    assert released.status == "released"
    assert released.actual_amount is None
    assert released.released_amount == released.reserved_amount
    assert released.reason == "Reservation expired: not reconciled within 60s."


def test_postgres_budget_ledger_rejects_unknown_reservation(postgres_dsn) -> None:
    async def ops(ledger):
        with pytest.raises(KeyError, match="bres_missing"):
            await ledger.reconcile(
                reservation_id="bres_missing",
                actual_amount=Decimal("0.01"),
            )
        with pytest.raises(KeyError, match="bres_missing"):
            await ledger.release(reservation_id="bres_missing", reason="unused")
        return True

    assert _run(postgres_dsn, ops) is True
