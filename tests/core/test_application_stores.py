"""Environment-selected application stores, shared pools, and the SQLite guard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest

from cayu import (
    ApplicationStores,
    KnowledgeAccessScope,
    KnowledgeQuery,
    SessionQuery,
    TaskCreate,
    TaskQuery,
    configured_database_url,
    open_application_stores,
)
from cayu.storage.targets import (
    PostgresRequiredError,
    SessionStoreBackend,
    SessionStoreTargetError,
    configured_database_pool_max,
    require_sqlite_store_allowed,
)

_SCOPE = KnowledgeAccessScope(allowed_namespaces=["project:app"])


@pytest.fixture(autouse=True)
def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        "CAYU_DATABASE_DIRECT_URL",
        "CAYU_DATABASE_POOL_MAX",
        "CAYU_REQUIRE_POSTGRES",
    ):
        monkeypatch.delenv(name, raising=False)


def _sqlite_store_constructors(path: Path):
    from cayu import (
        SQLiteAgentSnapshotStore,
        SQLiteAgentWorkContextStore,
        SQLiteBudgetLedger,
        SQLiteCollaborationStore,
        SQLiteEvalStore,
        SQLiteEventWatcherStore,
        SQLiteKnowledgeStore,
        SQLiteSessionStore,
        SQLiteTaskStore,
    )
    from cayu.browser_profiles import SQLiteBrowserProfileStore
    from cayu.memory.execution import SQLiteMemoryInterventionExecutionStore

    return {
        "SQLiteSessionStore": lambda: SQLiteSessionStore(path),
        "SQLiteTaskStore": lambda: SQLiteTaskStore(path),
        "SQLiteKnowledgeStore": lambda: SQLiteKnowledgeStore(path),
        "SQLiteEvalStore": lambda: SQLiteEvalStore(path),
        "SQLiteBudgetLedger": lambda: SQLiteBudgetLedger(path),
        "SQLiteEventWatcherStore": lambda: SQLiteEventWatcherStore(path),
        "SQLiteCollaborationStore": lambda: SQLiteCollaborationStore(path),
        "SQLiteAgentWorkContextStore": lambda: SQLiteAgentWorkContextStore(path),
        "SQLiteBrowserProfileStore": lambda: SQLiteBrowserProfileStore(path),
        "SQLiteMemoryInterventionExecutionStore": (
            lambda: SQLiteMemoryInterventionExecutionStore(path)
        ),
        "SQLiteAgentSnapshotStore": lambda: SQLiteAgentSnapshotStore(path),
    }


_SQLITE_STORE_NAMES = tuple(_sqlite_store_constructors(Path("unused.db")))


@pytest.mark.parametrize("store_name", _SQLITE_STORE_NAMES)
def test_required_postgres_refuses_every_sqlite_store_before_touching_disk(
    store_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")
    database = tmp_path / "state" / "cayu.db"

    with pytest.raises(PostgresRequiredError) as excinfo:
        _sqlite_store_constructors(database)[store_name]()

    message = str(excinfo.value)
    assert store_name in message
    assert "CAYU_REQUIRE_POSTGRES=1" in message
    assert "CAYU_DATABASE_URL" in message
    assert "cayu storage migrate" in message
    assert not database.parent.exists()


@pytest.mark.parametrize("value", [None, "", "0"])
def test_sqlite_stores_open_normally_without_the_postgres_requirement(
    value: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is not None:
        monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", value)
    from cayu import SQLiteSessionStore

    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "cayu.db")
        try:
            assert (await store.list_sessions(SessionQuery())).sessions == []
        finally:
            await store.close()

    asyncio.run(scenario())
    require_sqlite_store_allowed("AnyStore")


@pytest.mark.parametrize("value", ["true", "yes", "2"])
def test_unrecognized_postgres_requirement_fails_closed(
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", value)
    from cayu import SQLiteSessionStore

    with pytest.raises(PostgresRequiredError, match="must be 1"):
        SQLiteSessionStore(tmp_path / "cayu.db")


def test_default_application_stores_use_sqlite_at_the_absolute_path(tmp_path: Path) -> None:
    from cayu import SQLiteKnowledgeStore, SQLiteSessionStore, SQLiteTaskStore

    database = tmp_path / "data" / "cayu.db"

    async def scenario() -> None:
        async with open_application_stores(
            configured_database_url(),
            sqlite_path=database,
            knowledge_scope=_SCOPE,
        ) as stores:
            assert isinstance(stores, ApplicationStores)
            assert stores.backend is SessionStoreBackend.SQLITE
            assert stores.connection_budget == 0
            assert isinstance(stores.session_store, SQLiteSessionStore)
            assert isinstance(stores.task_store, SQLiteTaskStore)
            assert isinstance(stores.knowledge_store, SQLiteKnowledgeStore)
            assert stores.session_store.path == database
            assert stores.task_store.path == database
            await stores.task_store.create_task(TaskCreate(task_id="local", type="job"))
            result = await stores.knowledge_store.search(KnowledgeQuery(text="nothing"))
            assert result.hits == []
        assert database.is_file()

    asyncio.run(scenario())


def test_optional_application_stores_follow_the_selected_capabilities(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with open_application_stores(
            None,
            sqlite_path=tmp_path / "cayu.db",
            tasks=False,
        ) as stores:
            assert stores.task_store is None
            assert stores.knowledge_store is None

    asyncio.run(scenario())


def test_application_sqlite_path_must_be_absolute() -> None:
    with pytest.raises(SessionStoreTargetError, match="absolute"):
        open_application_stores(None, sqlite_path="data/cayu.db")


def test_application_stores_accept_an_absolute_sqlite_database_url(tmp_path: Path) -> None:
    selected = tmp_path / "selected.db"

    async def scenario() -> None:
        async with open_application_stores(
            f"sqlite://{quote(str(selected))}",
            sqlite_path=tmp_path / "default.db",
        ) as stores:
            assert stores.backend is SessionStoreBackend.SQLITE
            assert stores.session_store.path == selected.resolve()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "non-empty"),
        ("   ", "non-empty"),
        ("mysql://admin:do-not-print@db.example/app", "unsupported database URL scheme"),
        ("sqlite:relative.db", "absolute SQLite URL"),
    ],
)
def test_application_stores_reject_unusable_database_urls(
    url: str,
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(SessionStoreTargetError, match=message) as excinfo:
        open_application_stores(url, sqlite_path=tmp_path / "cayu.db")
    assert "do-not-print" not in str(excinfo.value)
    assert not (tmp_path / "cayu.db").exists()


def test_required_postgres_rejects_the_sqlite_default_before_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")
    database = tmp_path / "data" / "cayu.db"

    with pytest.raises(PostgresRequiredError, match="CAYU_DATABASE_URL"):
        open_application_stores(configured_database_url(), sqlite_path=database)
    assert not database.parent.exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 5), ("1", 1), (" 12 ", 12)],
)
def test_pool_max_defaults_to_five_and_reads_the_environment(
    value: str | None,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if value is not None:
        monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", value)
    assert configured_database_pool_max() == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "five", "2.5"])
def test_pool_max_rejects_non_positive_integers(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", value)
    with pytest.raises(SessionStoreTargetError, match="CAYU_DATABASE_POOL_MAX"):
        configured_database_pool_max()


def test_direct_database_url_must_be_postgres(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError, match="CAYU_DATABASE_DIRECT_URL"):
        open_application_stores(
            "postgresql://app@db.example/app",
            sqlite_path=tmp_path / "cayu.db",
            direct_database_url=f"sqlite://{tmp_path / 'other.db'}",
        )


def test_postgres_stores_share_one_lazy_pool_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cayu import PostgresKnowledgeStore, PostgresSessionStore, PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", "3")
    monkeypatch.setenv("CAYU_DATABASE_URL", "postgresql://app@127.0.0.1:1/unreachable")
    stores = open_application_stores(
        configured_database_url(),
        sqlite_path=tmp_path / "cayu.db",
        knowledge_scope=_SCOPE,
    )

    assert stores.backend is SessionStoreBackend.POSTGRES
    assert isinstance(stores.session_store, PostgresSessionStore)
    assert isinstance(stores.task_store, PostgresTaskStore)
    assert isinstance(stores.knowledge_store, PostgresKnowledgeStore)
    pools = {
        id(store._pool)
        for store in (stores.session_store, stores.task_store, stores.knowledge_store)
    }
    assert len(pools) == 1
    assert stores.task_store._pool.max_size == 3
    assert stores.pool_max_size == 3
    assert stores.connection_budget == 4
    for store in (stores.session_store, stores.task_store, stores.knowledge_store):
        assert store._owns_pool is False
        assert store._schema_mode is SchemaMode.VALIDATE
    assert stores.task_store._task_admission_listener_conninfo == configured_database_url()
    assert not (tmp_path / "cayu.db").exists()
    asyncio.run(stores.close())


def test_diagnostic_inspection_builds_read_only_store_owned_pools(tmp_path: Path) -> None:
    from cayu.storage._diagnostic_inspection import diagnostic_store_inspection

    with diagnostic_store_inspection():
        stores = open_application_stores(
            "postgresql://app@127.0.0.1:1/unreachable",
            sqlite_path=tmp_path / "cayu.db",
            knowledge_scope=_SCOPE,
        )
    for store in (stores.session_store, stores.task_store, stores.knowledge_store):
        assert store._owns_pool is True
        assert store._read_only is True
    asyncio.run(stores.close())


def _named(url: str, application_name: str) -> str:
    """Tag a connection URL so pg_stat_activity attributes its connections."""

    return f"{url}?application_name={quote(application_name, safe='')}"


@pytest.fixture(scope="module")
def migrated_postgres_url(postgres_dsn: str, postgres_url: str) -> str:
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def create_schema() -> None:
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

    asyncio.run(create_schema())
    return postgres_url


async def _connections_by_application(dsn: str, *names: str) -> dict[str, int]:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as observer:
        cursor = await observer.execute(
            "SELECT application_name, count(*) FROM pg_stat_activity "
            "WHERE application_name = ANY(%s) GROUP BY application_name",
            (list(names),),
        )
        rows = await cursor.fetchall()
    counts = dict.fromkeys(names, 0)
    counts.update({name: count for name, count in rows})
    return counts


def test_shared_pool_task_admission_wakeups_arrive_through_listen(
    migrated_postgres_url: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        consumer = open_application_stores(
            _named(migrated_postgres_url, "cayu-consumer"),
            sqlite_path=tmp_path / "unused.db",
        )
        producer = open_application_stores(
            _named(migrated_postgres_url, "cayu-producer"),
            sqlite_path=tmp_path / "unused.db",
            tasks=True,
        )
        wakeup = None
        try:
            assert consumer.task_store is not None and producer.task_store is not None
            wakeup = await consumer.task_store._task_admission_wakeup((TaskQuery(type="job"),))
            assert wakeup is not None
            first_attempt = consumer.task_store._task_admission_listener_first_attempt
            assert first_attempt is not None
            await asyncio.wait_for(first_attempt.wait(), timeout=10)
            listener = consumer.task_store._task_admission_listener_connection
            assert listener is not None
            listener_pid = listener.info.backend_pid
            assert listener_pid not in {
                connection.info.backend_pid for connection in consumer.task_store._pool._pool
            }

            hinted = asyncio.create_task(wakeup.wait(30.0, None))
            await asyncio.sleep(0)
            # Another process's store publishes; only LISTEN can deliver the hint.
            await producer.task_store.create_task(
                TaskCreate(task_id=f"remote-{uuid4().hex}", type="job")
            )
            assert await asyncio.wait_for(hinted, timeout=10) is False
            claimed = await consumer.task_store.claim_task("worker", TaskQuery(type="job"))
            assert claimed is not None
        finally:
            if wakeup is not None:
                wakeup.close()
            await consumer.close()
            await producer.close()

    asyncio.run(scenario())


def test_direct_database_url_carries_only_the_listen_connection(
    migrated_postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pooled_name = f"cayu-pooled-{uuid4().hex[:8]}"
    direct_name = f"cayu-direct-{uuid4().hex[:8]}"
    monkeypatch.setenv(
        "CAYU_DATABASE_URL",
        _named(migrated_postgres_url, pooled_name),
    )
    monkeypatch.setenv(
        "CAYU_DATABASE_DIRECT_URL",
        _named(migrated_postgres_url, direct_name),
    )

    async def scenario() -> None:
        stores = open_application_stores(
            configured_database_url(),
            sqlite_path=tmp_path / "unused.db",
        )
        wakeup = None
        try:
            assert stores.task_store is not None
            await stores.session_store.list_sessions(SessionQuery())
            wakeup = await stores.task_store._task_admission_wakeup((TaskQuery(type="job"),))
            first_attempt = stores.task_store._task_admission_listener_first_attempt
            assert first_attempt is not None
            await asyncio.wait_for(first_attempt.wait(), timeout=10)
            assert stores.task_store._task_admission_listener_connection is not None

            counts = await _connections_by_application(
                migrated_postgres_url, pooled_name, direct_name
            )
            assert counts[direct_name] == 1
            assert 1 <= counts[pooled_name] <= 5
        finally:
            if wakeup is not None:
                wakeup.close()
            await stores.close()
        assert await _connections_by_application(
            migrated_postgres_url, pooled_name, direct_name
        ) == {pooled_name: 0, direct_name: 0}

    asyncio.run(scenario())


def test_one_process_stays_within_the_pool_and_listener_budget(
    migrated_postgres_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application_name = f"cayu-budget-{uuid4().hex[:8]}"
    monkeypatch.setenv("CAYU_DATABASE_POOL_MAX", "2")
    monkeypatch.setenv(
        "CAYU_DATABASE_URL",
        _named(migrated_postgres_url, application_name),
    )

    async def scenario() -> None:
        stores = open_application_stores(
            configured_database_url(),
            sqlite_path=tmp_path / "unused.db",
            knowledge_scope=_SCOPE,
        )
        assert stores.connection_budget == 3
        observed: list[int] = []
        stop = asyncio.Event()

        async def sample() -> None:
            while not stop.is_set():
                counts = await _connections_by_application(migrated_postgres_url, application_name)
                observed.append(counts[application_name])
                await asyncio.sleep(0.01)

        wakeup = None
        sampler = asyncio.create_task(sample())
        try:
            assert stores.task_store is not None and stores.knowledge_store is not None
            wakeup = await stores.task_store._task_admission_wakeup((TaskQuery(type="job"),))
            first_attempt = stores.task_store._task_admission_listener_first_attempt
            assert first_attempt is not None
            await asyncio.wait_for(first_attempt.wait(), timeout=10)
            prefix = uuid4().hex
            await asyncio.gather(
                *(
                    stores.task_store.create_task(TaskCreate(task_id=f"{prefix}-{index}", type="x"))
                    for index in range(24)
                ),
                *(stores.session_store.list_sessions(SessionQuery()) for _ in range(12)),
                *(stores.knowledge_store.search(KnowledgeQuery(text="budget")) for _ in range(12)),
                *(stores.task_store.list_tasks(TaskQuery(type="x")) for _ in range(12)),
            )
        finally:
            stop.set()
            await sampler
            if wakeup is not None:
                wakeup.close()
            await stores.close()
        assert observed
        assert max(observed) <= stores.connection_budget
        assert max(observed) >= 2

    asyncio.run(scenario())
