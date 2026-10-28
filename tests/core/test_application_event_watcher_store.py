"""The durable event watcher store ``open_application_stores`` builds beside its session store."""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.events import Event, EventType
from cayu.observability.watchers import (
    EventWatcher,
    EventWatcherContext,
    InMemoryEventWatcherStore,
)
from cayu.sessions import InMemorySessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.application import ApplicationStores, open_application_stores

_UNREACHABLE = "postgresql://app@127.0.0.1:1/unreachable"
_SESSION_ID = "sess_watched"


@pytest.fixture(autouse=True)
def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        "CAYU_DATABASE_DIRECT_URL",
        "CAYU_DATABASE_POOL_MAX",
        "CAYU_REQUIRE_POSTGRES",
    ):
        monkeypatch.delenv(name, raising=False)


def _app(stores: ApplicationStores, **kwargs) -> CayuApp:
    return CayuApp(session_store=stores.session_store, enable_logging=False, **kwargs)


async def _append(stores: ApplicationStores, number: int) -> None:
    await stores.session_store.append_event(
        _SESSION_ID,
        Event(
            type=EventType.BUDGET_LIMIT_REACHED,
            session_id=_SESSION_ID,
            agent_name="assistant",
            environment_name="local-dev",
            payload={"number": number},
        ),
    )


def _watcher(handled: list[int]) -> EventWatcher:
    async def handler(context: EventWatcherContext) -> None:
        handled.append(context.record.event.payload["number"])

    return EventWatcher(
        name="budget-alerts",
        query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
        handler=handler,
    )


def _deliver_across_a_restart(open_stores) -> tuple[list[int], list[int]]:
    """Deliver two events, restart on the same database, then deliver one more."""

    before: list[int] = []
    after: list[int] = []

    async def before_restart() -> None:
        async with open_stores() as stores:
            await stores.session_store.create(
                RunRequest(
                    session_id=_SESSION_ID,
                    agent_name="assistant",
                    environment_name="local-dev",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await _append(stores, 1)
            await _append(stores, 2)
            app = _app(stores)
            assert app.event_watcher_store is stores.event_watcher_store
            await app.run_event_watchers([_watcher(before)])

    async def after_restart() -> None:
        async with open_stores() as stores:
            await _append(stores, 3)
            await _app(stores).run_event_watchers([_watcher(after)])

    asyncio.run(before_restart())
    asyncio.run(after_restart())
    return before, after


def test_sqlite_watchers_do_not_replay_old_events_after_a_restart(tmp_path: Path) -> None:
    from cayu.storage import SQLiteEventWatcherStore

    database = tmp_path / "data" / "cayu.db"

    def open_stores() -> ApplicationStores:
        stores = open_application_stores(None, sqlite_path=database)
        assert isinstance(stores.event_watcher_store, SQLiteEventWatcherStore)
        assert stores.event_watcher_store.path == database
        assert "event_watchers=True" in repr(stores)
        return stores

    before, after = _deliver_across_a_restart(open_stores)

    assert before == [1, 2]
    assert after == [3]


def test_an_app_supplied_watcher_store_wins_and_other_stores_keep_the_in_memory_default(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with open_application_stores(None, sqlite_path=tmp_path / "cayu.db") as stores:
            own = InMemoryEventWatcherStore()
            assert _app(stores, event_watcher_store=own).event_watcher_store is own
        assert type(CayuApp(enable_logging=False).event_watcher_store) is (
            InMemoryEventWatcherStore
        )
        unrelated = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        assert type(unrelated.event_watcher_store) is InMemoryEventWatcherStore

    asyncio.run(scenario())


def test_the_watcher_store_can_be_left_out(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_application_stores(
            None, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        ) as stores:
            assert stores.event_watcher_store is None
            assert "event_watchers=False" in repr(stores)
            assert type(_app(stores).event_watcher_store) is InMemoryEventWatcherStore

    asyncio.run(scenario())
    with pytest.raises(TypeError, match="event_watchers must be a bool"):
        open_application_stores(
            None,
            sqlite_path=tmp_path / "cayu.db",
            event_watchers="yes",  # ty: ignore[invalid-argument-type]
        )


def test_closing_the_stores_closes_the_watcher_store(tmp_path: Path) -> None:
    stores = open_application_stores(None, sqlite_path=tmp_path / "cayu.db")
    watcher_store = stores.event_watcher_store
    assert watcher_store is not None
    asyncio.run(stores.close())
    with pytest.raises(Exception, match="closed"):
        asyncio.run(watcher_store.load_state("after-close"))


def test_a_recorded_watcher_store_is_dropped_with_its_session_store() -> None:
    from cayu.observability import _store_watchers

    store = InMemorySessionStore()
    key = id(store)
    watcher_store = InMemoryEventWatcherStore()
    _store_watchers.attach_session_store_event_watcher_store(store, watcher_store)
    assert _store_watchers.session_store_event_watcher_store(store) is watcher_store
    assert key in _store_watchers._WATCHER_STORES._records

    del store
    gc.collect()

    assert key not in _store_watchers._WATCHER_STORES._records
    with pytest.raises(TypeError, match="must be an EventWatcherStore"):
        _store_watchers.attach_session_store_event_watcher_store(
            InMemorySessionStore(),
            object(),  # ty: ignore[invalid-argument-type]
        )


def test_diagnostic_inspection_builds_no_watcher_store(tmp_path: Path) -> None:
    from cayu.storage._diagnostic_inspection import diagnostic_store_inspection

    missing = tmp_path / "missing" / "cayu.db"
    with diagnostic_store_inspection() as inspection:
        stores = open_application_stores(None, sqlite_path=missing)
        assert stores.event_watcher_store is None
        asyncio.run(stores.close())
        postgres = open_application_stores(_UNREACHABLE, sqlite_path=missing)
        assert postgres.event_watcher_store is None
        asyncio.run(postgres.close())
    inspection.verify()
    assert not missing.exists()


def test_postgres_watcher_store_shares_the_pool_without_connecting(tmp_path: Path) -> None:
    from cayu.storage import PostgresEventWatcherStore, PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    stores = open_application_stores(_UNREACHABLE, sqlite_path=tmp_path / "cayu.db")
    watcher_store = stores.event_watcher_store
    session_store = stores.session_store
    assert isinstance(watcher_store, PostgresEventWatcherStore)
    assert isinstance(session_store, PostgresSessionStore)
    assert watcher_store._pool is session_store._pool
    assert watcher_store._owns_pool is False
    assert watcher_store._schema_mode is SchemaMode.VALIDATE
    # The watcher store adds no connections: the pool plus the task listener.
    assert stores.connection_budget == (stores.pool_max_size or 0) + 1
    assert _app(stores).event_watcher_store is watcher_store
    assert not (tmp_path / "cayu.db").exists()
    asyncio.run(stores.close())


def test_postgres_watchers_do_not_replay_old_events_after_a_restart(
    migrated_postgres_url: str, tmp_path: Path
) -> None:
    from cayu.storage import PostgresEventWatcherStore

    def open_stores() -> ApplicationStores:
        stores = open_application_stores(migrated_postgres_url, sqlite_path=tmp_path / "cayu.db")
        assert isinstance(stores.event_watcher_store, PostgresEventWatcherStore)
        return stores

    before, after = _deliver_across_a_restart(open_stores)

    assert before == [1, 2]
    assert after == [3]
    assert not (tmp_path / "cayu.db").exists()
