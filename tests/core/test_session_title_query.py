from __future__ import annotations

import asyncio

import pytest

from cayu import Message
from cayu.sessions import InMemorySessionStore, RunRequest, SessionIdentity, SessionQuery
from cayu.storage import SQLiteSessionStore


async def seed_title(store, session_id, title, **kwargs):
    return await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "Inspect")],
            title=title,
            **kwargs,
        ),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )


async def check_title_query(store):
    await seed_title(store, "one", "Quarterly 100% review_A")
    await seed_title(store, "two", "Quarterly 100% review_B")
    await seed_title(store, "three", "Other work", metadata={"note": "quarterly 100% review_"})
    await seed_title(store, "four", "Other 100x reviewZ")
    first = await store.list_sessions(
        SessionQuery(q="QUARTERLY", limit=1, include_total_count=True, order_by="created_at_asc")
    )
    assert [row.id for row in first.sessions] == ["one"]
    assert first.total_count == 2
    assert first.next_cursor is not None
    second = await store.list_sessions(
        SessionQuery(
            q="QUARTERLY",
            limit=1,
            include_total_count=True,
            order_by="created_at_asc",
            cursor=first.next_cursor,
        )
    )
    assert [row.id for row in second.sessions] == ["two"]
    assert second.next_cursor is None
    assert second.total_count == 2
    literal = await store.list_sessions(SessionQuery(q="100% review_"))
    assert {row.id for row in literal.sessions} == {"one", "two"}
    original = await store.list_sessions(SessionQuery(q="assistant"))
    assert {row.id for row in original.sessions} == {"one", "two", "three", "four"}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_title_search_retains_native_filter_count_and_cursor_semantics(backend, tmp_path):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db")
        )
        try:
            await check_title_query(store)
        finally:
            if backend == "sqlite":
                await store.close()
        if backend == "sqlite":
            reopened = SQLiteSessionStore(tmp_path / "sessions.db")
            try:
                page = await reopened.list_sessions(SessionQuery(q="QUARTERLY"))
                assert {row.id for row in page.sessions} == {"one", "two"}
            finally:
                await reopened.close()

    asyncio.run(run())


def test_postgres_title_search_retains_native_query_semantics(postgres_dsn):
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await check_title_query(store)
        finally:
            await store.close()

    asyncio.run(run())
