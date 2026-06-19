from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import (
    CayuApp,
    Event,
    EventQuery,
    EventType,
    EventWatcher,
    EventWatcherContext,
    EventWatcherDeliveryStatus,
    RunRequest,
    SQLiteEventWatcherStore,
)
from cayu.runtime import InMemoryEventWatcherStore, InMemorySessionStore
from cayu.runtime.sessions import EventRecord, SessionIdentity
from cayu.storage.migrations import SchemaMode

_POSTGRES_TABLES = (
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


class CountingSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.query_event_limits: list[int] = []

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        self.query_event_limits.append(EventQuery().limit if query is None else query.limit)
        return await super().query_events(query)


async def _create_session(store: InMemorySessionStore, session_id: str = "sess_1") -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            environment_name="local-dev",
            messages=[],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


async def _append_event(
    store: InMemorySessionStore,
    *,
    session_id: str = "sess_1",
    event_type: EventType | str = EventType.BUDGET_LIMIT_REACHED,
    agent_name: str = "assistant",
    payload: dict | None = None,
) -> Event:
    event = Event(
        type=event_type,
        session_id=session_id,
        agent_name=agent_name,
        environment_name="local-dev",
        payload={} if payload is None else payload,
    )
    await store.append_event(session_id, event)
    return event


def test_event_watcher_handles_matching_events_once() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        await _append_event(session_store, event_type=EventType.MODEL_STARTED)
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )

        first_results = await app.run_event_watchers([watcher])
        second_results = await app.run_event_watchers([watcher])
        return first, handled, first_results, second_results

    event, handled, first_results, second_results = asyncio.run(run())
    assert handled == [event.id]
    assert first_results[0].deliveries[0].status is EventWatcherDeliveryStatus.SUCCEEDED
    assert first_results[0].deliveries[0].event_id == event.id
    assert second_results[0].deliveries == []


def test_event_watcher_fetches_matching_events_in_batches() -> None:
    async def run():
        session_store = CountingSessionStore()
        await _create_session(session_store)
        events = [
            await _append_event(session_store, payload={"number": number})
            for number in range(3)
        ]
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            batch_size=3,
        )

        results = await app.run_event_watchers([watcher], limit=3)
        return events, handled, session_store.query_event_limits, results

    events, handled, query_limits, results = asyncio.run(run())
    assert handled == [event.id for event in events]
    assert query_limits == [3]
    assert [delivery.event_id for delivery in results[0].deliveries] == [
        event.id for event in events
    ]


def test_event_watcher_large_batch_uses_capped_event_query_pages() -> None:
    async def run():
        session_store = CountingSessionStore()
        await _create_session(session_store)
        events = [
            await _append_event(session_store, payload={"number": number})
            for number in range(5001)
        ]
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            batch_size=5001,
        )

        results = await app.run_event_watchers([watcher], limit=5001)
        return events, handled, session_store.query_event_limits, results

    events, handled, query_limits, results = asyncio.run(run())
    assert handled == [event.id for event in events]
    assert query_limits == [5000, 1]
    assert [delivery.event_id for delivery in results[0].deliveries] == [
        event.id for event in events
    ]


def test_event_watcher_retries_failed_event_before_later_events() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        seen: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            seen.append(context.record.event.id)
            if context.record.event.id == first.id and context.attempt == 1:
                raise RuntimeError("temporary email failure")

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )

        failed = await app.run_event_watchers([watcher], limit=10)
        retried = await app.run_event_watchers([watcher], limit=10)
        return first, second, seen, failed, retried

    first, second, seen, failed, retried = asyncio.run(run())
    assert seen == [first.id, first.id, second.id]
    assert failed[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert failed[0].deliveries[0].attempt == 1
    assert [delivery.status for delivery in retried[0].deliveries] == [
        EventWatcherDeliveryStatus.SUCCEEDED,
        EventWatcherDeliveryStatus.SUCCEEDED,
    ]
    assert [delivery.event_id for delivery in retried[0].deliveries] == [first.id, second.id]


def test_event_watcher_dead_letters_after_max_attempts_and_unblocks_cursor() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)
            if context.record.event.id == first.id:
                raise RuntimeError("permanent webhook failure")

        app = CayuApp(
            session_store=session_store,
            event_watcher_store=InMemoryEventWatcherStore(),
            enable_logging=False,
        )
        watcher = EventWatcher(
            name="budget-webhook",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
            max_attempts=2,
        )

        first_failure = await app.run_event_watchers([watcher], limit=10)
        dead_letter_then_success = await app.run_event_watchers([watcher], limit=10)
        state = await app.event_watcher_store.load_state("budget-webhook")
        return first, second, handled, first_failure, dead_letter_then_success, state

    first, second, handled, first_failure, second_run, state = asyncio.run(run())
    assert handled == [first.id, first.id, second.id]
    assert first_failure[0].deliveries[0].status is EventWatcherDeliveryStatus.FAILED
    assert [delivery.status for delivery in second_run[0].deliveries] == [
        EventWatcherDeliveryStatus.DEAD_LETTERED,
        EventWatcherDeliveryStatus.SUCCEEDED,
    ]
    assert [delivery.event_id for delivery in second_run[0].deliveries] == [first.id, second.id]
    assert state.cursor_sequence == second_run[0].deliveries[-1].event_sequence
    assert state.dead_lettered_count == 1


def test_event_watcher_active_lease_blocks_duplicate_processing() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        record_event = await _append_event(session_store)
        store = InMemoryEventWatcherStore()
        records = await session_store.query_events(EventQuery(limit=1))
        first_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=300,
        )
        second_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=300,
        )
        return record_event, first_claim, second_claim

    event, first_claim, second_claim = asyncio.run(run())
    assert first_claim is not None
    assert first_claim.event_id == event.id
    assert second_claim is None


def test_event_watcher_expired_lease_can_be_reclaimed() -> None:
    async def run():
        now = {"value": datetime(2026, 1, 1, tzinfo=UTC)}

        def clock() -> datetime:
            return now["value"]

        session_store = InMemorySessionStore()
        await _create_session(session_store)
        record_event = await _append_event(session_store)
        store = InMemoryEventWatcherStore(clock=clock)
        records = await session_store.query_events(EventQuery(limit=1))
        first_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=10,
        )

        now["value"] = now["value"] + timedelta(seconds=11)
        second_claim = await store.claim_event(
            watcher_name="budget-email",
            record=records[0],
            lease_seconds=10,
        )
        return record_event, first_claim, second_claim

    event, first_claim, second_claim = asyncio.run(run())
    assert first_claim is not None
    assert first_claim.event_id == event.id
    assert second_claim is not None
    assert second_claim.event_id == event.id
    assert second_claim.attempt == 2
    assert second_claim.claim_id != first_claim.claim_id


def test_sqlite_event_watcher_store_persists_cursor(tmp_path: Path) -> None:
    async def run():
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        db_path = tmp_path / "watchers.sqlite"
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )
        first_store = SQLiteEventWatcherStore(db_path)
        first_app = CayuApp(
            session_store=session_store,
            event_watcher_store=first_store,
            enable_logging=False,
        )
        first_result = await first_app.run_event_watchers([watcher], limit=1)
        await first_store.close()

        second_store = SQLiteEventWatcherStore(db_path)
        second_app = CayuApp(
            session_store=session_store,
            event_watcher_store=second_store,
            enable_logging=False,
        )
        second_result = await second_app.run_event_watchers([watcher], limit=10)
        state = await second_store.load_state("budget-email")
        await second_store.close()
        return first, second, handled, first_result, second_result, state

    first, second, handled, first_result, second_result, state = asyncio.run(run())
    assert handled == [first.id, second.id]
    assert [delivery.event_id for delivery in first_result[0].deliveries] == [first.id]
    assert [delivery.event_id for delivery in second_result[0].deliveries] == [second.id]
    assert state.cursor_sequence == second_result[0].deliveries[-1].event_sequence


async def _drop_postgres_tables(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _POSTGRES_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def test_postgres_event_watcher_store_persists_cursor(postgres_dsn: str) -> None:
    async def run():
        from cayu import PostgresEventWatcherStore

        await _drop_postgres_tables(postgres_dsn)
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        first = await _append_event(session_store, payload={"number": 1})
        second = await _append_event(session_store, payload={"number": 2})
        handled: list[str] = []

        async def handler(context: EventWatcherContext) -> None:
            handled.append(context.record.event.id)

        watcher = EventWatcher(
            name="budget-email",
            query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
            handler=handler,
        )
        first_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            first_app = CayuApp(
                session_store=session_store,
                event_watcher_store=first_store,
                enable_logging=False,
            )
            first_result = await first_app.run_event_watchers([watcher], limit=1)
        finally:
            await first_store.close()

        second_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            second_app = CayuApp(
                session_store=session_store,
                event_watcher_store=second_store,
                enable_logging=False,
            )
            second_result = await second_app.run_event_watchers([watcher], limit=10)
            state = await second_store.load_state("budget-email")
        finally:
            await second_store.close()
        return first, second, handled, first_result, second_result, state

    first, second, handled, first_result, second_result, state = asyncio.run(run())
    assert handled == [first.id, second.id]
    assert [delivery.event_id for delivery in first_result[0].deliveries] == [first.id]
    assert [delivery.event_id for delivery in second_result[0].deliveries] == [second.id]
    assert state.cursor_sequence == second_result[0].deliveries[-1].event_sequence


def test_postgres_event_watcher_store_serializes_first_claim(postgres_dsn: str) -> None:
    async def run():
        from cayu import PostgresEventWatcherStore

        await _drop_postgres_tables(postgres_dsn)
        session_store = InMemorySessionStore()
        await _create_session(session_store)
        event = await _append_event(session_store)
        records = await session_store.query_events(EventQuery(limit=1))
        first_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        second_store = PostgresEventWatcherStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            claim_results = await asyncio.gather(
                first_store.claim_event(
                    watcher_name="budget-email",
                    record=records[0],
                    lease_seconds=300,
                ),
                second_store.claim_event(
                    watcher_name="budget-email",
                    record=records[0],
                    lease_seconds=300,
                ),
            )
        finally:
            await first_store.close()
            await second_store.close()
        return event, claim_results

    event, claim_results = asyncio.run(run())
    claims = [claim for claim in claim_results if claim is not None]
    blocked = [claim for claim in claim_results if claim is None]
    assert len(claims) == 1
    assert len(blocked) == 1
    assert claims[0].event_id == event.id


def test_event_watcher_rejects_cursor_in_query() -> None:
    with pytest.raises(ValueError, match="after_sequence"):
        EventWatcher(
            name="invalid",
            query=EventQuery(after_sequence=10),
            handler=lambda _context: None,
        )
