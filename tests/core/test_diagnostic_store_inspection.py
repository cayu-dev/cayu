from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from cayu import Message, RunRequest, SQLiteSessionStore
from cayu.runtime.sessions import SessionIdentity
from cayu.storage import _sqlite_support
from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.event_watchers import SQLiteEventWatcherStore
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.work_context_sqlite import SQLiteAgentWorkContextStore


def test_sqlite_diagnostic_inspection_rejects_incomplete_wal_sidecars(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(database)
        await store.close()

    asyncio.run(initialize())
    wal = Path(f"{database}-wal")
    wal.write_bytes(b"incomplete-wal-canary")

    with (
        diagnostic_store_inspection(),
        pytest.raises(sqlite3.OperationalError, match="sidecars are incomplete"),
    ):
        _sqlite_support.connect_read_only_inspection(database)

    assert wal.read_bytes() == b"incomplete-wal-canary"
    assert not Path(f"{database}-shm").exists()


def test_ordinary_read_only_session_store_sees_writer_started_later(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(database)
        await store.close()

    asyncio.run(initialize())
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()

    async def exercise() -> None:
        reader = SQLiteSessionStore(
            database,
            schema_mode=SchemaMode.VALIDATE,
            read_only=True,
        )
        writer = SQLiteSessionStore(database)
        assert reader.service_durability.value == "read_only"
        try:
            assert await reader.load_state("sess_reader_started_first") is None
            await writer.create(
                RunRequest(
                    session_id="sess_reader_started_first",
                    agent_name="writer",
                    messages=[Message.text("user", "created after reader")],
                ),
                identity=SessionIdentity(provider_name="test", model="model"),
            )

            loaded = await reader.load_state("sess_reader_started_first")
            assert loaded is not None
            assert loaded.id == "sess_reader_started_first"
        finally:
            await writer.close()
            await reader.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "store_type",
    (
        SQLiteAgentWorkContextStore,
        SQLiteBudgetLedger,
        SQLiteEvalStore,
        SQLiteEventWatcherStore,
        SQLiteKnowledgeStore,
        SQLiteSessionStore,
        SQLiteTaskStore,
    ),
)
def test_diagnostic_inspection_forces_every_builtin_sqlite_store_read_only(
    store_type: type,
    tmp_path: Path,
) -> None:
    database = tmp_path / "runtime.sqlite"

    async def initialize() -> None:
        store = SQLiteSessionStore(database)
        await store.close()

    asyncio.run(initialize())
    before_bytes = database.read_bytes()
    before_stat = database.stat()
    before_entries = tuple(sorted(item.name for item in tmp_path.iterdir()))

    with diagnostic_store_inspection() as inspection:
        store = store_type(database)
        assert store._connection.execute("PRAGMA query_only").fetchone()[0] == 1
        if isinstance(store, (SQLiteSessionStore, SQLiteTaskStore)):
            assert store.service_durability.value == "durable"
        inspection.verify()

    asyncio.run(store.close())
    after_stat = database.stat()
    assert database.read_bytes() == before_bytes
    assert (after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert tuple(sorted(item.name for item in tmp_path.iterdir())) == before_entries


@pytest.mark.parametrize(
    "store_type",
    (
        SQLiteAgentWorkContextStore,
        SQLiteBudgetLedger,
        SQLiteEvalStore,
        SQLiteEventWatcherStore,
        SQLiteKnowledgeStore,
        SQLiteSessionStore,
        SQLiteTaskStore,
    ),
)
def test_diagnostic_inspection_allows_process_private_in_memory_sqlite_stores(
    store_type: type,
) -> None:
    with diagnostic_store_inspection() as inspection:
        store = store_type(":memory:")
        assert store._connection.execute("PRAGMA query_only").fetchone()[0] == 0
        database = store._connection.execute("PRAGMA database_list").fetchone()
        assert database[1:] == ("main", "")
        if isinstance(store, (SQLiteSessionStore, SQLiteTaskStore)):
            assert store.service_durability.value == "development"
        inspection.verify()

    asyncio.run(store.close())


def test_diagnostic_inspection_allows_control_plane_read_only_memory_eval_store() -> None:
    with diagnostic_store_inspection() as inspection:
        store = SQLiteEvalStore(
            ":memory:",
            schema_mode=SchemaMode.VALIDATE,
            read_only=True,
        )
        assert store._connection.execute("PRAGMA query_only").fetchone()[0] == 0
        assert store._connection.execute("PRAGMA database_list").fetchone()[1:] == (
            "main",
            "",
        )
        inspection.verify()

    asyncio.run(store.close())
