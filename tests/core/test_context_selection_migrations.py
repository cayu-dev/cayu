"""Forward migration and startup qualification for native selection fences."""

import sqlite3
from contextlib import closing

import pytest
from tests.core.test_context_selection_exclusion import scenario

from cayu.collaboration._contracts import ExactMatch
from cayu.sessions._context_selection_fence import (
    _CONTEXT_SELECTION_AUTHORITY,
    ContextViewSelectionExcluded,
)
from cayu.storage import migrations as schema
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["sqlite", "postgres"])
async def database(request, tmp_path):
    stores = []
    if request.param == "sqlite":
        path = tmp_path / "selection.sqlite"

        async def execute(statement):
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(statement)
                connection.commit()

        def create(mode):
            store = SQLiteSessionStore(path, schema_mode=mode)
            stores.append(store)
            return store

    else:
        import psycopg

        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")
        # postgres_dsn is an isolated, disposable per-module test database.
        async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute("DROP SCHEMA public CASCADE")
            await connection.execute("CREATE SCHEMA public")

        async def execute(statement):
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                await connection.execute(statement)

        def create(mode):
            store = PostgresSessionStore(dsn, schema_mode=mode)
            stores.append(store)
            return store

    try:
        yield create, execute
    finally:
        for store in reversed(stores):
            await store.close()


@pytest.mark.parametrize("first", ["selection", "exclusion"])
async def test_revision_107_upgrade_preserves_view_and_exact_decision(database, monkeypatch, first):
    create, _ = database
    with monkeypatch.context() as historical:
        historical.setattr(
            schema, "REVISIONS", tuple(rev for rev in schema.REVISIONS if rev.revision <= 107)
        )
        store = create(schema.SchemaMode.CREATE)
        _, request = await scenario(store)
        source = await store.load(request.source_session_id)
        await store.close()
    store = create(schema.SchemaMode.MIGRATE)
    assert await store.load(request.source_session_id) == source
    if first == "selection":
        selected = await store.select_context_view(request)
    decision = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert decision.state == ("selected" if first == "selection" else "excluded")
    await store.close()
    for mode in (schema.SchemaMode.MIGRATE, schema.SchemaMode.VALIDATE):
        store = create(mode)
        found = await store.read_context_view_selection_decision(request)
        assert isinstance(found, ExactMatch) and found.receipt == decision
        assert (
            await store._exclude_context_view_selection(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
            == decision
        )
        if first == "selection":
            assert await store.select_context_view(request) == selected
        else:
            with pytest.raises(ContextViewSelectionExcluded):
                await store.select_context_view(request)
        assert await store.load(request.source_session_id) == source
        await store.close()


@pytest.mark.parametrize("mode", list(schema.SchemaMode))
@pytest.mark.parametrize("damage", ["table", "column", "index", "wrong_index"])
async def test_current_revision_rejects_damaged_selection_fence(database, mode, damage):
    create, execute = database
    store = create(schema.SchemaMode.CREATE)
    assert await store.load("absent") is None
    await store.close()
    if damage == "table":
        await execute("DROP TABLE cayu_context_selection_exclusions")
    elif damage == "column":
        await execute("ALTER TABLE cayu_context_selection_exclusions DROP COLUMN decision_json")
    else:
        await execute("DROP INDEX idx_context_selection_exclusions_owner")
        if damage == "wrong_index":
            await execute(
                "CREATE INDEX idx_context_selection_exclusions_owner "
                "ON cayu_context_selection_exclusions(source_session_id)"
            )
    with pytest.raises(schema.SchemaError, match="Context selection fence schema"):
        await create(mode).load("absent")
