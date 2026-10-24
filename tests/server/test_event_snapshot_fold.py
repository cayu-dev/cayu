"""Native exact event snapshots shared by control-plane usage consumers."""

import asyncio

import pytest

from cayu import Event, EventType, Message, RunRequest
from cayu.sessions import InMemorySessionStore, SessionIdentity
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.storage import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_snapshot_fold_pages_and_releases_after_consumer_failure(tmp_path, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "snapshot.sqlite")
        )
        try:
            await store.create(
                RunRequest(
                    session_id="one", agent_name="agent", messages=[Message.text("user", "go")]
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            for index in range(5):
                await store.append_event(
                    "one",
                    Event(
                        id=f"call-{index}",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="one",
                        tool_name="tool",
                    ),
                )
            pages = []
            through = await store.fold_event_snapshot(
                EventQuery(limit=2), pages.append, max_bytes=100000
            )
            records = [record for page in pages for record in page]
            assert len(records) == 5
            assert [record.sequence for record in records] == sorted(
                record.sequence for record in records
            )
            assert through == records[-1].sequence
            assert all(len(page) <= 2 for page in pages)

            class Stop(Exception):
                pass

            partial = []

            def fail(page):
                partial.extend(page)
                if len(partial) == 2:
                    raise Stop()

            with pytest.raises(Stop):
                await store.fold_event_snapshot(EventQuery(limit=1), fail, max_bytes=100000)
            assert len(partial) == 2
            oversized = []
            with pytest.raises(EventQueryResultTooLarge):
                await store.fold_event_snapshot(EventQuery(limit=2), oversized.extend, max_bytes=1)
            assert oversized == []
            descending = []
            assert (
                await store.fold_event_snapshot(
                    EventQuery(limit=2, order_by=EventOrder.SEQUENCE_DESC),
                    descending.extend,
                    max_bytes=100000,
                    max_pages=1,
                )
                == through
            )
            assert [record.sequence for record in descending] == [
                record.sequence for record in records[-2:]
            ][::-1]
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


def test_sqlite_snapshot_holds_visibility_and_drains_cancelled_worker(tmp_path):
    import threading

    async def run():
        store = SQLiteSessionStore(tmp_path / "owned.sqlite")
        await store.create(
            RunRequest(session_id="one", agent_name="agent", messages=[Message.text("user", "go")]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )

        async def append(index):
            await store.append_event(
                "one",
                Event(
                    id=f"call-{index}",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="one",
                    tool_name="tool",
                ),
            )

        await append(0)
        for cancel in (False, True):
            entered = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()
            rows = []

            def consume(page, rows=rows, entered=entered, loop=loop, release=release):
                rows.extend(page)
                if not entered.is_set():
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(10)

            pending = asyncio.create_task(
                store.fold_event_snapshot(EventQuery(limit=1), consume, max_bytes=100000)
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                await append(1 if not cancel else 2)
                if cancel:
                    pending.cancel()
                    await asyncio.sleep(0.05)
                    assert pending.cancelling() == 1 and not pending.done()
                release.set()
                if cancel:
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert pending.cancelled() and pending.cancelling() == 1
                else:
                    through = await pending
                    assert len(rows) == 1 and through == rows[0].sequence
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
        rows = []
        await store.fold_event_snapshot(EventQuery(limit=1), rows.extend, max_bytes=100000)
        assert len(rows) == 3
        await store.close()

    asyncio.run(run())


def test_postgres_snapshot_visibility_between_pages_and_task_cancellation(
    postgres_dsn, monkeypatch
):
    from tests.core.postgres_contention_support import drop_cayu_tables

    from cayu.storage import PostgresSessionStore
    from cayu.storage import _postgres_session_queries as queries
    from cayu.storage.migrations import SchemaMode

    async def run():
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
        )
        await store.create(
            RunRequest(session_id="one", agent_name="agent", messages=[Message.text("user", "go")]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )

        async def append(index):
            await store.append_event(
                "one",
                Event(
                    id=f"call-{index}",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="one",
                    tool_name="tool",
                ),
            )

        for index in range(5):
            await append(index)
        original = queries._query_events_bounded_snapshot
        try:
            for cancel in (False, True):
                entered, release = asyncio.Event(), asyncio.Event()

                async def gated(
                    cur, query, *, max_bytes, exact_snapshot, entered=entered, release=release
                ):
                    records = await original(
                        cur, query, max_bytes=max_bytes, exact_snapshot=exact_snapshot
                    )
                    if exact_snapshot and query.limit == 2 and not entered.is_set():
                        entered.set()
                        await release.wait()
                    return records

                monkeypatch.setattr(queries, "_query_events_bounded_snapshot", gated)
                rows = []
                pending = asyncio.create_task(
                    store.fold_event_snapshot(EventQuery(limit=2), rows.extend, max_bytes=100000)
                )
                try:
                    await asyncio.wait_for(entered.wait(), 10)
                    if cancel:
                        pending.cancel()
                        assert pending.cancelling() == 1
                        with pytest.raises(asyncio.CancelledError):
                            await pending
                        assert pending.cancelled() and pending.cancelling() == 1
                    else:
                        await append(5)
                        release.set()
                        through = await asyncio.wait_for(pending, 10)
                        assert len(rows) == 5 and through == rows[-1].sequence
                finally:
                    release.set()
                    await asyncio.gather(pending, return_exceptions=True)
            monkeypatch.setattr(queries, "_query_events_bounded_snapshot", original)
            rows = []
            await store.fold_event_snapshot(EventQuery(limit=2), rows.extend, max_bytes=100000)
            assert len(rows) == 6
        finally:
            await store.close()

    asyncio.run(run())
