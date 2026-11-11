"""Cross-session event cursors on Postgres never skip an event that commits later.

``cayu_events.sequence`` is drawn at INSERT but becomes visible at COMMIT, so a
transaction can commit a higher sequence while another one still holds a lower
one. A reader that pages the global stream by ``after_sequence`` must not return
the higher one until the lower one is settled, or its cursor moves past it and
it never returns that event.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

pytestmark = pytest.mark.postgres


def _store(dsn: str):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    return PostgresSessionStore(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)


async def _reset(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")


async def _create_sessions(store, *session_ids: str) -> None:
    for session_id in session_ids:
        await store.create(
            RunRequest(
                agent_name="agent",
                session_id=session_id,
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )


async def _insert_event(conn, session_id: str, event_id: str, order: int) -> None:
    event = Event(
        id=event_id,
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        agent_name="agent",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    document = event.model_dump(mode="json")
    await conn.execute(
        "INSERT INTO cayu_events (session_id, session_order, event_id, event_type, "
        "timestamp, agent_name, payload, event) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            session_id,
            order,
            event_id,
            str(event.type),
            event.timestamp,
            event.agent_name,
            json.dumps(document["payload"]),
            json.dumps(document),
        ),
    )


async def _head(store) -> int:
    records = await store.query_events(EventQuery(order_by=EventOrder.SEQUENCE_DESC, limit=1))
    return records[0].sequence if records else 0


async def _wait_for(predicate, *, timeout: float = 30.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await predicate()
        if result:
            return result
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("the condition did not hold in time")
        await asyncio.sleep(0.05)


def _ids(records) -> list[str]:
    return [record.event.id for record in records]


def test_a_later_commit_with_a_lower_sequence_is_not_skipped(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        early = await psycopg.AsyncConnection.connect(postgres_dsn)
        late = await psycopg.AsyncConnection.connect(postgres_dsn)
        try:
            await _create_sessions(store, "sess_early", "sess_late")
            cursor = await _head(store)

            # The issue's interleaving: `early` takes its transaction id first, `late`
            # draws the lower sequence, and `early` commits while `late` is still open.
            await early.execute("SELECT pg_current_xact_id()")
            await late.execute("SELECT pg_current_xact_id()")
            await _insert_event(late, "sess_late", "late", 1000)
            await _insert_event(early, "sess_early", "early", 1000)
            await early.commit()

            page = await store.query_events(EventQuery(after_sequence=cursor, limit=10))
            # Returning `early` alone would move the cursor past `late`.
            assert _ids(page) == []
            # A single-session read has no such hazard and sees the committed event.
            own = await store.query_events(
                EventQuery(session_id="sess_early", after_sequence=cursor, limit=10)
            )
            assert _ids(own) == ["early"]

            await late.commit()

            async def both():
                records = await store.query_events(EventQuery(after_sequence=cursor, limit=10))
                return records if len(records) == 2 else None

            assert _ids(await _wait_for(both)) == ["late", "early"]
        finally:
            await early.close()
            await late.close()
            await store.close()

    asyncio.run(scenario())


def test_a_rolled_back_writer_stops_holding_readers_back(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        aborted = await psycopg.AsyncConnection.connect(postgres_dsn)
        committed = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        try:
            await _create_sessions(store, "sess_a", "sess_b")
            cursor = await _head(store)
            await _insert_event(aborted, "sess_a", "aborted", 1000)
            await _insert_event(committed, "sess_b", "kept", 1000)
            assert _ids(await store.query_events(EventQuery(after_sequence=cursor))) == []
            await aborted.rollback()

            async def kept():
                return await store.query_events(EventQuery(after_sequence=cursor, limit=10))

            assert _ids(await _wait_for(kept)) == ["kept"]
        finally:
            await aborted.close()
            await committed.close()
            await store.close()

    asyncio.run(scenario())


def test_a_writer_holds_one_reservation_however_many_events_it_inserts(
    postgres_dsn: str,
) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        writer = await psycopg.AsyncConnection.connect(postgres_dsn)
        try:
            await _create_sessions(store, "sess_bulk")
            for order in range(1000, 1200):
                await _insert_event(writer, "sess_bulk", f"event_{order}", order)
            cursor = await writer.execute(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            )
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1
            await writer.commit()
        finally:
            await writer.close()
            await store.close()

    asyncio.run(scenario())


def test_unrelated_advisory_locks_do_not_hold_readers_back(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        holder = await psycopg.AsyncConnection.connect(postgres_dsn)
        writer = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        try:
            await _create_sessions(store, "sess_a")
            cursor = await _head(store)
            # Advisory locks with small keys in both key forms, as other code takes them.
            await holder.execute("SELECT pg_advisory_xact_lock(0), pg_advisory_xact_lock(0, 0)")
            await _insert_event(writer, "sess_a", "visible", 1000)

            async def visible():
                return await store.query_events(EventQuery(after_sequence=cursor, limit=10))

            assert _ids(await _wait_for(visible, timeout=5.0)) == ["visible"]
            await holder.rollback()
        finally:
            await holder.close()
            await writer.close()
            await store.close()

    asyncio.run(scenario())


def test_a_rolled_back_savepoint_reserves_again_on_the_next_insert(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        writer = await psycopg.AsyncConnection.connect(postgres_dsn)
        committed = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        try:
            await _create_sessions(store, "sess_a", "sess_b")
            cursor = await _head(store)
            await writer.execute("SAVEPOINT attempt")
            await _insert_event(writer, "sess_a", "discarded", 1000)
            await writer.execute("ROLLBACK TO SAVEPOINT attempt")
            await _insert_event(writer, "sess_a", "pending", 1001)
            reservations = await writer.execute(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            )
            row = await reservations.fetchone()
            assert row is not None and row[0] == 1
            await _insert_event(committed, "sess_b", "after", 1000)
            # The writer's open reservation still holds the committed event back.
            assert _ids(await store.query_events(EventQuery(after_sequence=cursor))) == []
            await writer.commit()

            async def both():
                records = await store.query_events(EventQuery(after_sequence=cursor, limit=10))
                return records if len(records) == 2 else None

            assert _ids(await _wait_for(both)) == ["pending", "after"]
        finally:
            await writer.close()
            await committed.close()
            await store.close()

    asyncio.run(scenario())


def test_bounded_reads_take_the_bound_before_their_snapshot(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        open_writer = await psycopg.AsyncConnection.connect(postgres_dsn)
        committed = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        try:
            await _create_sessions(store, "sess_a", "sess_b")
            cursor = await _head(store)
            await _insert_event(open_writer, "sess_a", "lower", 1000)
            await _insert_event(committed, "sess_b", "higher", 1000)
            query = EventQuery(after_sequence=cursor, limit=10)
            assert _ids(await store.query_events_bounded(query, max_bytes=1_000_000)) == []
            [many] = await store.query_events_bounded_many([query], max_bytes=1_000_000)
            assert not isinstance(many, Exception) and _ids(many) == []
            await open_writer.commit()

            async def both():
                records = await store.query_events_bounded(query, max_bytes=1_000_000)
                return records if len(records) == 2 else None

            assert _ids(await _wait_for(both)) == ["lower", "higher"]
        finally:
            await open_writer.close()
            await committed.close()
            await store.close()

    asyncio.run(scenario())


def test_the_bound_refuses_a_repeatable_read_snapshot(postgres_dsn: str) -> None:
    import psycopg

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        await _head(store)  # creates the schema
        await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            with pytest.raises(psycopg.errors.RaiseException, match="READ COMMITTED"):
                await conn.execute("SELECT cayu_event_visible_sequence('cayu_events'::regclass)")

    asyncio.run(scenario())


def test_the_high_water_cannot_be_pointed_at_another_table(postgres_dsn: str) -> None:
    import psycopg
    from psycopg.conninfo import make_conninfo

    role = "cayu_event_cursor_reader"

    async def scenario() -> None:
        await _reset(postgres_dsn)
        store = _store(postgres_dsn)
        await _head(store)  # creates the schema
        await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as admin:
            await admin.execute(f"DROP ROLE IF EXISTS {role}")
            await admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'reader'")
            await admin.execute(
                "CREATE TABLE secret_ledger (sequence BIGINT GENERATED ALWAYS AS IDENTITY)"
            )
            await admin.execute("INSERT INTO secret_ledger DEFAULT VALUES")
            await admin.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        try:
            reader_dsn = make_conninfo(postgres_dsn, user=role, password="reader")
            async with await psycopg.AsyncConnection.connect(reader_dsn, autocommit=True) as reader:
                await reader.execute("CREATE TEMP TABLE pg_class (oid oid, relname name)")
                await reader.execute(
                    "INSERT INTO pg_temp.pg_class VALUES ('secret_ledger'::regclass, 'cayu_events')"
                )
                with pytest.raises(psycopg.errors.RaiseException, match="only cayu_events"):
                    await reader.execute(
                        "SELECT cayu_event_sequence_high_water('secret_ledger'::regclass)"
                    )
        finally:
            async with await psycopg.AsyncConnection.connect(
                postgres_dsn, autocommit=True
            ) as admin:
                await admin.execute("DROP TABLE IF EXISTS secret_ledger")
                await admin.execute(f"REVOKE ALL ON SCHEMA public FROM {role}")
                await admin.execute(f"DROP ROLE IF EXISTS {role}")

    asyncio.run(scenario())


def test_cursor_reads_work_when_the_database_defaults_to_repeatable_read(
    postgres_dsn: str,
) -> None:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict

    database = conninfo_to_dict(postgres_dsn)["dbname"]

    async def set_default(level: str | None) -> None:
        async with await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True) as admin:
            statement = (
                sql.SQL("ALTER DATABASE {} RESET default_transaction_isolation")
                if level is None
                else sql.SQL("ALTER DATABASE {} SET default_transaction_isolation = {}").format(
                    sql.Identifier(str(database)), sql.Literal(level)
                )
            )
            if level is None:
                statement = statement.format(sql.Identifier(str(database)))
            await admin.execute(statement)

    async def scenario() -> None:
        await _reset(postgres_dsn)
        setup = _store(postgres_dsn)
        await _create_sessions(setup, "sess_a", "sess_b")
        cursor = await _head(setup)
        await setup.close()
        await set_default("repeatable read")
        store = _store(postgres_dsn)
        writer = await psycopg.AsyncConnection.connect(postgres_dsn, autocommit=True)
        try:
            await _insert_event(writer, "sess_a", "first", 1000)
            await _insert_event(writer, "sess_b", "second", 1000)
            query = EventQuery(after_sequence=cursor, limit=10)

            async def both():
                records = await store.query_events(query)
                return records if len(records) == 2 else None

            assert _ids(await _wait_for(both)) == ["first", "second"]
            batched = await store.query_events(
                EventQuery(session_ids=("sess_a", "sess_b"), after_sequence=cursor, limit=10)
            )
            assert _ids(batched) == ["first", "second"]
            bounded = await store.query_events_bounded(query, max_bytes=1_000_000)
            assert _ids(bounded) == ["first", "second"]
            [many] = await store.query_events_bounded_many([query], max_bytes=1_000_000)
            assert not isinstance(many, Exception) and _ids(many) == ["first", "second"]
        finally:
            await writer.close()
            await store.close()
            await set_default(None)

    asyncio.run(scenario())
