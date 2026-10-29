"""Current environment activity through native store snapshots."""

from __future__ import annotations

import asyncio

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore


async def seed_environment_sessions(store, prefix):
    for suffix, environment in (("a", "dev"), ("b", "dev"), ("c", "prod"), ("d", None)):
        await store.create(
            RunRequest(
                session_id=f"{prefix}-{suffix}",
                agent_name="writer",
                environment_name=environment,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="p", model="m"),
        )
    await store.append_event(
        f"{prefix}-a", Event(session_id=f"{prefix}-a", type=EventType.MODEL_COMPLETED, payload={})
    )
    return await store.load(f"{prefix}-a")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_environment_activity_counts_and_timestamp_native(backend, tmp_path):
    from cayu.sessions.environment_activity import EnvironmentActivityQuery

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "activity.sqlite")
        )
        try:
            active = await seed_environment_sessions(store, "example")
            await store.update_metadata("example-a", {"note": "metadata update"})
            query = EnvironmentActivityQuery(environment_names=("prod", "dev", "empty"))
            snapshot = await store.aggregate_environment_activity(query)
            snapshot.validate_for_query(query)
            by_name = {row.environment_name: row for row in snapshot.environments}
            assert by_name["dev"].session_count == 2
            assert by_name["dev"].last_activity_at == active.last_activity_at
            assert by_name["prod"].session_count == 1
            assert by_name["empty"].session_count == 0
            assert by_name["empty"].last_activity_at is None
            assert snapshot.model_dump(mode="json")["environments"][1]["session_count"] == "2"
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "names", [(), ("dev", "dev"), ("",), (" dev",), (True,), tuple(f"e{i}" for i in range(501))]
)
def test_environment_activity_query_rejects_ambiguous_or_unbounded_names(names):
    from cayu.sessions.environment_activity import EnvironmentActivityQuery

    with pytest.raises((TypeError, ValueError)):
        EnvironmentActivityQuery(environment_names=names)


def test_environment_activity_sqlite_snapshot_is_batched_and_survives_restart(
    tmp_path, monkeypatch
):
    from cayu.sessions.environment_activity import EnvironmentActivityQuery
    from cayu.storage import _sqlite_session_queries

    async def run():
        path = tmp_path / "restart.sqlite"
        store = SQLiteSessionStore(path)
        try:
            await seed_environment_sessions(store, "restart")
            calls = []
            original = _sqlite_session_queries.aggregate_environment_activity

            async def observed(run_read, query):
                async def traced(fn):
                    def read(connection):
                        connection.set_trace_callback(calls.append)
                        try:
                            return fn(connection)
                        finally:
                            connection.set_trace_callback(None)

                    return await run_read(read)

                return await original(traced, query)

            monkeypatch.setattr(_sqlite_session_queries, "aggregate_environment_activity", observed)
            query = EnvironmentActivityQuery(
                environment_names=("dev", "prod", *(f"empty-{i}" for i in range(498)))
            )
            first = await store.aggregate_environment_activity(query)
            assert len(first.environments) == 500
            assert len([sql for sql in calls if "GROUP BY environment_name" in sql]) == 1
            assert not any("cayu_events" in sql for sql in calls)
            await store.close()
            store = SQLiteSessionStore(path)
            second = await store.aggregate_environment_activity(query)
            assert second.environments == first.environments
            await store.delete_session("restart-a")
            await store.create(
                RunRequest(
                    session_id="restart-a",
                    agent_name="writer",
                    environment_name="prod",
                    messages=[Message.text("user", "new")],
                ),
                identity=SessionIdentity(provider_name="p", model="m"),
            )
            current = await store.aggregate_environment_activity(query)
            assert [row.session_count for row in current.environments[:2]] == [1, 2]
        finally:
            await store.close()

    asyncio.run(run())


def test_environment_activity_cancellation_propagates_from_real_lock_wait():
    from cayu.sessions.environment_activity import EnvironmentActivityQuery

    async def run():
        store = InMemorySessionStore()
        async with store._lock:
            task = asyncio.create_task(
                store.aggregate_environment_activity(
                    EnvironmentActivityQuery(environment_names=("dev",))
                )
            )
            await asyncio.sleep(0)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
        result = await store.aggregate_environment_activity(
            EnvironmentActivityQuery(environment_names=("dev",))
        )
        assert result.environments[0].session_count == 0

    asyncio.run(run())


def test_postgres_environment_activity_is_exact(postgres_dsn):
    from uuid import uuid4

    from cayu.sessions.environment_activity import EnvironmentActivityQuery
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        scope = uuid4().hex
        query = EnvironmentActivityQuery(environment_names=(scope, scope + "-empty"))
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )
        try:
            for i in range(3):
                await store.create(
                    RunRequest(
                        session_id=f"{scope}-{i}",
                        agent_name="writer",
                        environment_name=scope,
                        messages=[Message.text("user", "hi")],
                    ),
                    identity=SessionIdentity(provider_name="p", model="m"),
                )
            activity_times = []
            for i in range(3):
                row = await store.load(f"{scope}-{i}")
                assert row is not None
                activity_times.append(row.last_activity_at)
            result = await store.aggregate_environment_activity(query)
            result.validate_for_query(query)
            assert result.environments[0].session_count == 3
            assert result.environments[0].last_activity_at == max(activity_times)
            assert result.environments[1].session_count == 0
        finally:
            await store.close()

    asyncio.run(run())
