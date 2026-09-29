from __future__ import annotations

import asyncio
import json
import threading

import pytest

from cayu.storage._validated_cache import validated_row_cache
from cayu.storage.sqlite import SQLiteSessionStore, _checkpoint_from_json


def test_validated_cache_detaches_and_invalidates_exact_content():
    calls = []

    @validated_row_cache
    def decode(value):
        calls.append(value)
        return json.loads(value)

    first = decode('{"items": [1]}')
    first["items"].append(2)
    assert decode('{"items": [1]}') == {"items": [1]}
    assert len(calls) == 1
    assert decode('{"items": [2]}') == {"items": [2]}
    assert len(calls) == 2
    for _ in range(2):
        with pytest.raises(ValueError):
            decode("invalid")
    assert len(calls) == 4


def test_checkpoint_cache_rejects_changed_invalid_content():
    source = '{"cache-regression": [1]}'
    _checkpoint_from_json(source)["cache-regression"].append(2)
    assert _checkpoint_from_json(source) == {"cache-regression": [1]}
    with pytest.raises(ValueError):
        _checkpoint_from_json('{"cache-regression": NaN}')


def test_validated_cache_is_bounded(monkeypatch):
    from cayu.storage import _validated_cache

    monkeypatch.setattr(_validated_cache, "_MAX_SOURCE_BYTES", 1024)
    monkeypatch.setattr(_validated_cache, "_MAX_ENTRIES", 2)
    calls = []

    @validated_row_cache
    def decode(value):
        calls.append(value)
        return value

    for value in ("a", "b", "c", "a"):
        assert decode(value) == value
    assert calls == ["a", "b", "c", "a"]
    for _ in range(2):
        decode("x" * 1024)
    assert len(calls) == 6


def test_read_pool_parallelism_and_cancelled_lease(sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path()))
            release = threading.Event()
            started = [threading.Event() for _ in range(4)]
            connections = []

            def blocked(index, connection):
                connections.append(connection)
                started[index].set()
                if not release.wait(5):
                    raise TimeoutError("reader was not released")
                return connection.execute("SELECT 1").fetchone()[0]

            owners = [
                resources.task(store._run_read(lambda conn, i=i: blocked(i, conn)))
                for i in range(4)
            ]
            try:
                for event in started:
                    assert await asyncio.to_thread(event.wait, 2)
                assert len({id(connection) for connection in connections}) == 4
                owners[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owners[0]
                follower = resources.task(
                    store._run_read(lambda conn: conn.execute("SELECT 2").fetchone()[0])
                )
                await asyncio.sleep(0.02)
                assert not follower.done()
                assert (
                    await store._run_write(lambda conn: conn.execute("SELECT 3").fetchone()[0]) == 3
                )
            finally:
                release.set()
                await asyncio.gather(*owners, return_exceptions=True)
            assert await follower == 2

    asyncio.run(scenario())


def test_session_cache_tracks_external_updates_without_timestamp_change(sqlite_resources):
    from cayu.sessions.base import RunRequest, SessionIdentity

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path()))
            session = await store.create(
                RunRequest(agent_name="agent", messages=[], metadata={"nested": [1]}),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            first = await store.load(session.id)
            assert first is not None
            first.metadata["nested"].append(2)
            second = await store.load(session.id)
            assert second is not None
            assert second.metadata["nested"] == [1]

            def update_metadata(connection, metadata):
                with connection:
                    connection.execute(
                        "UPDATE cayu_sessions SET metadata_json = ? WHERE id = ?",
                        (json.dumps(metadata), session.id),
                    ).close()

            await store._run_write(
                lambda conn: update_metadata(conn, second.metadata | {"nested": [3]})
            )
            updated = await store.load(session.id)
            assert updated is not None
            assert updated.updated_at == second.updated_at
            assert updated.metadata["nested"] == [3]
            await store._run_write(
                lambda conn: update_metadata(conn, second.metadata | {"nested": float("nan")})
            )
            with pytest.raises(ValueError):
                await store.load(session.id)

    asyncio.run(scenario())


def test_validated_cache_does_not_conflate_sql_scalar_types():
    calls = []

    @validated_row_cache
    def validate(value):
        calls.append(value)
        if type(value) is not int:
            raise ValueError("integer required")
        return value

    assert validate(1) == 1
    with pytest.raises(ValueError):
        validate(1.0)
    assert len(calls) == 2


@pytest.mark.parametrize("event_count", [1, 100])
def test_event_scans_avoid_redundant_copies_and_revalidate_content(
    sqlite_resources, monkeypatch, event_count
):
    from cayu._validation import MAX_DURABLE_JSON_INTEGER
    from cayu.events import Event
    from cayu.sessions.base import RunRequest, SessionIdentity

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path()))
            session = await store.create(
                RunRequest(agent_name="agent", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            events = [
                Event(
                    type="custom.scan",
                    session_id=session.id,
                    payload={"index": index, "nested": [{"value": "original"}]},
                )
                for index in range(event_count)
            ]
            await store.append_events(session.id, events)
            copies = []
            deepcopy_event = Event.__deepcopy__

            def count_copy(event, memo=None):
                copies.append(event.id)
                return deepcopy_event(event, memo)

            monkeypatch.setattr(Event, "__deepcopy__", count_copy)
            first = await store.load_events(session.id)
            assert first == events
            first[0].payload["nested"][0]["value"] = "changed"
            assert await store.load_events(session.id) == events
            # Freshly decoded events already belong to the caller. Copying
            # their payload and validation stamp again makes scans slower,
            # including repeated histories larger than the cache capacity.
            assert copies == []

            def corrupt_payload(connection):
                with connection:
                    connection.execute(
                        "UPDATE cayu_events SET payload_json = ? WHERE event_id = ?",
                        (json.dumps({"invalid": MAX_DURABLE_JSON_INTEGER + 1}), events[0].id),
                    ).close()

            await store._run_write(corrupt_payload)
            with pytest.raises(ValueError):
                await store.load_events(session.id)

    asyncio.run(scenario())
