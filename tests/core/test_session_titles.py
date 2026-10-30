"""Minimal native title projections do not hydrate whole sessions or events."""

import asyncio

import pytest

from cayu import RunRequest
from cayu.sessions.display import SessionTitleQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_title_batch_is_ordered_bounded_and_missing_safe(tmp_path, backend, monkeypatch):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "titles.sqlite")
        )
        try:
            for key, title in (("one", "First title"), ("two", None)):
                await store.create(
                    RunRequest(session_id=key, agent_name="writer", title=title, messages=[]),
                    identity=SessionIdentity(provider_name="p", model="m"),
                )

            async def forbidden(*args, **kwargs):
                raise AssertionError("title projection must not hydrate a session or its history")

            monkeypatch.setattr(store, "load", forbidden)
            monkeypatch.setattr(store, "load_events", forbidden)
            query = SessionTitleQuery(session_ids=("two", "missing", "one"))
            snapshot = await store.read_session_titles(query)
            snapshot.validate_for_query(query)
            assert [(row.session_id, row.title) for row in snapshot.sessions] == [
                ("two", None),
                ("missing", None),
                ("one", "First title"),
            ]
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


def test_title_projection_survives_sqlite_reopen_and_rejects_mutated_queries(tmp_path):
    async def run():
        path = tmp_path / "persistent.sqlite"
        store = SQLiteSessionStore(path)
        await store.create(
            RunRequest(
                session_id="persisted", agent_name="writer", title="Durable title", messages=[]
            ),
            identity=SessionIdentity(provider_name="p", model="m"),
        )
        await store.close()
        reopened = SQLiteSessionStore(path)
        try:
            assert (
                await reopened.read_session_titles(SessionTitleQuery(session_ids=("persisted",)))
            ).sessions[0].title == "Durable title"
            invalid = SessionTitleQuery(session_ids=("persisted",))
            invalid.session_ids = (True,)
            with pytest.raises(ValueError):
                await reopened.read_session_titles(invalid)
            for ids in ((), ("same", "same"), tuple(str(index) for index in range(501))):
                with pytest.raises(ValueError):
                    SessionTitleQuery(session_ids=ids)
        finally:
            await reopened.close()

    asyncio.run(run())


def test_title_projection_postgres_native_read(postgres_dsn):
    from uuid import uuid4

    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        key = f"artifact-title-{uuid4()}"
        try:
            await store.create(
                RunRequest(
                    session_id=key, agent_name="writer", title="Postgres title", messages=[]
                ),
                identity=SessionIdentity(provider_name="p", model="m"),
            )
            snapshot = await store.read_session_titles(
                SessionTitleQuery(session_ids=(key, "missing-artifact-producer"))
            )
            assert [(row.session_id, row.title) for row in snapshot.sessions] == [
                (key, "Postgres title"),
                ("missing-artifact-producer", None),
            ]
        finally:
            await store.close()

    asyncio.run(run())
