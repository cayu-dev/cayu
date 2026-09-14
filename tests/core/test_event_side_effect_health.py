from __future__ import annotations

# ruff: noqa: F811 - imported pytest fixtures
import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_session_store_shared_conformance import (
    _close_store,
    _open_store,
    _public_authority_alias_codec,
    conformance_postgres_dsn,  # noqa: F401
    session_store_case,  # noqa: F401
)

from cayu import Event, EventType, Message
from cayu.runtime import PersistedEventSideEffectQuery, RunRequest, SessionIdentity


@pytest.mark.usefixtures("session_store_case")
def test_health_lifecycle_and_inspection(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            empty = await store.get_persisted_event_side_effect_health()
            assert empty.outstanding_total == empty.delivered == 0
            assert empty.oldest_claimable_at is None
            session = await store.create(
                RunRequest(
                    messages=[Message.text("user", "hello")], agent_name="test", session_id="health"
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            events = [
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session.id,
                    payload={"private": "NEVER_EXPORT"},
                )
                for _ in range(6)
            ]
            for event in events:
                await store.append_event(session.id, event)
            claims = []
            for event in events[1:]:
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session.id, event_id=event.id
                )
                assert claim is not None
                claims.append(claim)
            await store.mark_persisted_event_side_effect_failed(
                claims[1], error="secret\npassword", max_attempts=3, retry_delay_seconds=0
            )
            again = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=events[2].id
            )
            await store.mark_persisted_event_side_effect_failed(
                again, error="secret", max_attempts=3, retry_delay_seconds=0
            )
            await store.mark_persisted_event_side_effect_failed(
                claims[2], error="secret", max_attempts=1, retry_delay_seconds=0
            )
            await store.mark_persisted_event_side_effect_delivered(claims[3])
            await store.mark_persisted_event_side_effect_failed(
                claims[4], error="secret", max_attempts=3, retry_delay_seconds=3600
            )
            health = await store.get_persisted_event_side_effect_health()
            assert (
                health.pending,
                health.leased_live,
                health.failed_retryable,
                health.failed_deferred,
                health.dead_lettered,
                health.delivered,
            ) == (1, 1, 2, 1, 1, 1)
            assert health.claimable_total == 2
            assert health.outstanding_total == 5
            assert health.repeatedly_failing == health.final_attempt_boundary == 1
            assert health.max_outstanding_attempts == 2
            assert health.oldest_claimable_age_seconds >= 0
            page = await store.query_persisted_event_side_effect_deliveries(
                PersistedEventSideEffectQuery()
            )
            assert len(page.deliveries) == 5
            assert (
                "secret" not in page.model_dump_json()
                and "NEVER_EXPORT" not in page.model_dump_json()
            )
            assert sum(row.claimable for row in page.deliveries) == 2
            assert (
                len(
                    (
                        await store.query_persisted_event_side_effect_deliveries(
                            PersistedEventSideEffectQuery(statuses={"failed"}, claimable_only=True)
                        )
                    ).deliveries
                )
                == 1
            )
            before = health.model_dump(
                exclude={
                    "observed_at",
                    "oldest_claimable_age_seconds",
                    "oldest_pending_age_seconds",
                    "oldest_failed_age_seconds",
                    "oldest_dead_letter_age_seconds",
                }
            )
            after = (await store.get_persisted_event_side_effect_health()).model_dump(
                exclude={
                    "observed_at",
                    "oldest_claimable_age_seconds",
                    "oldest_pending_age_seconds",
                    "oldest_failed_age_seconds",
                    "oldest_dead_letter_age_seconds",
                }
            )
            assert before == after
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_health_pagination_beyond_one_batch(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            for name in ["Z-page", "a-page"]:
                session = await store.create(
                    RunRequest(
                        messages=[Message.text("user", "hello")], agent_name="test", session_id=name
                    ),
                    identity=SessionIdentity(provider_name="fake", model="fake"),
                )
                for _ in range(505):
                    await store.append_event(
                        session.id, Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
                    )
            seen = set()
            ordered = []
            cursor = None
            while True:
                result = await store.query_persisted_event_side_effect_deliveries(
                    PersistedEventSideEffectQuery(limit=137, cursor=cursor)
                )
                for row in result.deliveries:
                    key = row.session_id, row.event_id
                    assert key not in seen
                    seen.add(key)
                    ordered.append(key)
                cursor = result.next_cursor
                if cursor is None:
                    break
                with pytest.raises(ValueError):
                    await store.query_persisted_event_side_effect_deliveries(
                        PersistedEventSideEffectQuery(cursor=cursor, claimable_only=True)
                    )
            assert ordered == sorted(ordered)
            assert len(seen) == 1010
            assert (await store.get_persisted_event_side_effect_health()).pending == 1010
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_skew_and_expired_lease_memory():
    from cayu.runtime import InMemorySessionStore
    from cayu.sessions.base import PersistedEventSideEffectDelivery

    async def run():
        now = datetime.now(UTC)
        store = InMemorySessionStore()
        store._ownership_clock = lambda: now
        store._persisted_event_side_effect_deliveries[("s", "e")] = (
            PersistedEventSideEffectDelivery(
                session_id="s",
                event_id="e",
                event_sequence=1,
                status="pending",
                updated_at=now + timedelta(days=1),
            )
        )
        assert (
            await store.get_persisted_event_side_effect_health()
        ).oldest_pending_age_seconds == 0
        store._persisted_event_side_effect_deliveries[("s", "e")] = (
            PersistedEventSideEffectDelivery(
                session_id="s",
                event_id="e",
                event_sequence=1,
                status="leased",
                lease_expires_at=now - timedelta(seconds=3),
            )
        )
        health = await store.get_persisted_event_side_effect_health()
        assert health.leased_expired == health.claimable_total == 1
        assert health.oldest_claimable_age_seconds == 3

    asyncio.run(run())


def test_health_expired_claim_and_acknowledgement_race(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    messages=[Message.text("user", "hello")],
                    agent_name="test",
                    session_id="lease-health",
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            event = Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id, lease_seconds=0.02
            )
            assert claim is not None
            await asyncio.sleep(0.04)
            health = await store.get_persisted_event_side_effect_health()
            assert health.leased_expired == health.claimable_total == 1
            assert health.leased_live == 0
            replacement = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id
            )
            assert replacement is not None
            live = await store.get_persisted_event_side_effect_health()
            assert live.max_outstanding_attempts == 2
            assert live.final_attempt_boundary == 0  # attempt two is still in progress
            from cayu.runtime import PersistedEventSideEffectClaimLost

            with pytest.raises(PersistedEventSideEffectClaimLost):
                await store.mark_persisted_event_side_effect_delivered(claim)
            snapshot, _ = await asyncio.gather(
                store.get_persisted_event_side_effect_health(),
                store.mark_persisted_event_side_effect_delivered(replacement),
            )
            assert snapshot.leased_live + snapshot.delivered == 1
            assert snapshot.claimable_total == 0
            assert (await store.get_persisted_event_side_effect_health()).delivered == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_health_uses_aggregate_and_covering_index(tmp_path):
    from cayu.runtime.event_side_effect_health import health_sql
    from cayu.storage import SQLiteSessionStore, _sqlite_support

    async def run():
        store = SQLiteSessionStore(tmp_path / "health.sqlite")
        try:
            await store.get_persisted_event_side_effect_health()
            statements = []

            def traced(connection):
                connection.set_trace_callback(statements.append)

            await store._run_read(traced)
            await store.get_persisted_event_side_effect_health()
            selected = [sql for sql in statements if "cayu_persisted_event_side_effects" in sql]
            assert len(selected) == 1
            assert "SUM(CASE" in selected[0]
            assert "cayu_events" not in selected[0]

            def explain(connection):
                return [
                    row[3]
                    for row in connection.execute(
                        "EXPLAIN QUERY PLAN " + health_sql("?"),
                        (_sqlite_support.format_datetime(datetime.now(UTC)),),
                    )
                ]

            plan = await store._run_read(explain)
            assert any("COVERING INDEX idx_cayu_side_effect_health" in line for line in plan)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_health_index_migration_preserves_rows(tmp_path):
    import sqlite3

    from cayu.storage import SQLiteSessionStore
    from cayu.storage.migrations import SchemaMode, SchemaState, validate

    path = tmp_path / "migration.sqlite"

    async def run():
        store = SQLiteSessionStore(path)
        session = await store.create(
            RunRequest(
                messages=[Message.text("user", "hello")], agent_name="test", session_id="migration"
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.append_event(
            session.id, Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
        )
        await _close_store(store)
        with sqlite3.connect(path) as conn:
            conn.execute("DROP INDEX idx_cayu_side_effect_health")
            conn.execute("DROP INDEX idx_cayu_side_effect_outstanding")
            conn.execute("DELETE FROM cayu_schema_migrations WHERE revision = 86")
        store = SQLiteSessionStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            assert (await store.get_persisted_event_side_effect_health()).pending == 1

            def indexes(connection):
                return {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA index_list('cayu_persisted_event_side_effects')"
                    )
                }

            assert {
                "idx_cayu_side_effect_health",
                "idx_cayu_side_effect_outstanding",
            } <= await store._run_read(indexes)
            validate(
                SchemaState(revision=86, compatible_from=85), app_latest=85, app_min_supported=85
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_postgres_health_index_migration_preserves_rows(conformance_postgres_dsn):
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = PostgresSessionStore(
            conformance_postgres_dsn, public_authority_alias_codec=_public_authority_alias_codec()
        )
        try:
            session = await store.create(
                RunRequest(
                    messages=[Message.text("user", "hello")],
                    agent_name="test",
                    session_id="pg-health-migration",
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_event(
                session.id, Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
            )
            before = (await store.get_persisted_event_side_effect_health()).pending
            async with store._connection() as conn, conn.cursor() as cur:
                await cur.execute("DROP INDEX idx_cayu_side_effect_health")
                await cur.execute("DROP INDEX idx_cayu_side_effect_outstanding")
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision = 86")
                await conn.commit()
        finally:
            await _close_store(store)
        store = PostgresSessionStore(
            conformance_postgres_dsn,
            schema_mode=SchemaMode.MIGRATE,
            public_authority_alias_codec=_public_authority_alias_codec(),
        )
        try:
            assert (await store.get_persisted_event_side_effect_health()).pending == before
            async with store._connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'cayu_persisted_event_side_effects'"
                )
                indexes = {row[0] for row in await cur.fetchall()}
                assert {
                    "idx_cayu_side_effect_health",
                    "idx_cayu_side_effect_outstanding",
                } <= indexes
        finally:
            await _close_store(store)

    asyncio.run(run())
