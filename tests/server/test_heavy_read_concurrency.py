"""Heavy dashboard reads stay bounded and laned, so light reads answer while they run."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from typing import Annotated, Any

import httpx
import pytest
from fastapi import Body, FastAPI
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _start

from cayu.server import ServerConfig, create_server
from cayu.server._heavy_reads import (
    DEFAULT_HEAVY_READ_CONCURRENCY,
    ClientDisconnectMiddleware,
    HeavyReadLimiter,
)
from cayu.storage._postgres_base import bulk_connection_limit
from cayu.storage._read_lanes import ReadLane, bulk_reads, current_read_lane
from cayu.storage.sqlite import SQLiteSessionStore

__all__ = ["conformance_postgres_dsn", "session_store_case"]

# A light read normally answers in milliseconds. Without lanes it waits for a
# held bulk read, which these tests never release until the light read is done.
LIGHT_READ_SECONDS = 5.0
_LONG_SCAN = (
    "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5000000000) "
    "SELECT count(*) FROM n"
)


def _client(app: Any) -> httpx.AsyncClient:
    server = create_server(app, config=ServerConfig.local_development())
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test")


class _Holders:
    """Count bulk reads holding a store connection, and release them together."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.release = threading.Event()

    def enter(self) -> None:
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)

    def leave(self) -> None:
        with self.lock:
            self.running -= 1


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _hold_sqlite_reader(holders: _Holders):
    def hold(connection: sqlite3.Connection) -> None:
        holders.enter()
        try:
            while not holders.release.wait(0.01):
                connection.execute("SELECT 1").fetchone()
        finally:
            holders.leave()

    return hold


@pytest.mark.store_semantics
def test_bulk_reads_never_hold_the_connections_light_reads_use(session_store_case):
    store_kind = session_store_case[0]
    if store_kind == "memory":
        pytest.skip("The in-memory store has no connections to separate.")

    async def scenario():
        store: Any = await _open_store(session_store_case)
        holders = _Holders()
        try:
            await _start(_app(store), "light", "writer")
            if store_kind == "sqlite":
                assert isinstance(store, SQLiteSessionStore)
                capacity = 2

                async def bulk_read() -> None:
                    with bulk_reads():
                        await store._run_read(_hold_sqlite_reader(holders))

            else:
                capacity = bulk_connection_limit(store._pool.max_size)
                released = asyncio.Event()

                async def bulk_read() -> None:
                    with bulk_reads():
                        async with store._connection() as conn:
                            await conn.execute("SELECT 1")
                            holders.enter()
                            try:
                                await released.wait()
                            finally:
                                holders.leave()

            # More bulk reads than the store has connections in all.
            bulk = [asyncio.create_task(bulk_read()) for _ in range(8)]
            try:
                await _wait_until(lambda: holders.running == capacity)
                started = time.monotonic()
                session = await asyncio.wait_for(store.load("light"), LIGHT_READ_SECONDS)
                assert session is not None and session.id == "light"
                assert time.monotonic() - started < LIGHT_READ_SECONDS
                assert holders.running == capacity
            finally:
                holders.release.set()
                if store_kind != "sqlite":
                    released.set()
                await asyncio.gather(*bulk)
            assert holders.peak == capacity
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def test_heavy_routes_run_a_bounded_number_while_a_session_read_answers(tmp_path, monkeypatch):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        holders = _Holders()
        lanes: list[ReadLane] = []
        try:
            app = _app(store)
            await _start(app, "light", "writer")

            async def held_tool_usage(*args, **kwargs):
                lanes.append(current_read_lane())
                return await store._run_read(_hold_sqlite_reader(holders))

            monkeypatch.setattr(store, "tool_usage_totals", held_tool_usage)
            async with _client(app) as client:
                heavy = [asyncio.create_task(client.get("/api/tools/usage")) for _ in range(5)]
                try:
                    await _wait_until(lambda: holders.running == DEFAULT_HEAVY_READ_CONCURRENCY)
                    await asyncio.sleep(0.1)
                    # The rest wait for an admission slot, not for a connection.
                    assert len(lanes) == DEFAULT_HEAVY_READ_CONCURRENCY
                    started = time.monotonic()
                    response = await asyncio.wait_for(
                        client.get("/api/sessions/light"), LIGHT_READ_SECONDS
                    )
                    elapsed = time.monotonic() - started
                    assert response.status_code == 200, response.text
                    assert elapsed < LIGHT_READ_SECONDS
                    # A plain list joins no overviews, so it needs no heavy slot.
                    listing = await asyncio.wait_for(
                        client.get("/api/sessions", params={"limit": 10}), LIGHT_READ_SECONDS
                    )
                    assert listing.status_code == 200, listing.text
                    assert len(lanes) == DEFAULT_HEAVY_READ_CONCURRENCY
                finally:
                    holders.release.set()
                    # The stub returns no totals, so how each request ends is not checked.
                    await asyncio.gather(*heavy, return_exceptions=True)
            assert holders.peak == DEFAULT_HEAVY_READ_CONCURRENCY
            assert lanes == [ReadLane.BULK] * 5
        finally:
            await store.close()

    asyncio.run(scenario())


def _http_scope(method: str, path: str, *, body: bytes = b"") -> dict[str, Any]:
    headers = [(b"host", b"test")]
    if body:
        headers += [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }


class _Connection:
    """An ASGI client that sends its request, then disconnects when told to."""

    def __init__(self, body: bytes = b"") -> None:
        self._body = body
        self._sent_body = False
        self.disconnect = asyncio.Event()
        self.messages: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        if not self._sent_body:
            self._sent_body = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


def test_disconnect_interrupts_a_heavy_sqlite_read_and_frees_its_slot(tmp_path, monkeypatch):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        holders = _Holders()
        interrupted = threading.Event()
        try:
            app = _app(store)
            await _start(app, "light", "writer")
            original = store.tool_usage_totals

            def until_interrupted(connection: sqlite3.Connection) -> None:
                holders.enter()
                try:
                    # One long statement, as a real aggregate scan is: it runs for
                    # minutes unless SQLite interrupts it.
                    connection.execute(_LONG_SCAN).fetchone()
                except sqlite3.OperationalError:
                    interrupted.set()
                    raise
                finally:
                    holders.leave()

            async def held_tool_usage(*args, **kwargs):
                if not interrupted.is_set():
                    return await store._run_read(until_interrupted)
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, "tool_usage_totals", held_tool_usage)
            server = create_server(app, config=ServerConfig.local_development())
            connections = [_Connection() for _ in range(DEFAULT_HEAVY_READ_CONCURRENCY)]
            requests = [
                asyncio.create_task(
                    server(_http_scope("GET", "/api/tools/usage"), item.receive, item.send)
                )
                for item in connections
            ]
            await _wait_until(lambda: holders.running == DEFAULT_HEAVY_READ_CONCURRENCY)
            # Let both statements start; an interrupt before a statement is a no-op.
            await asyncio.sleep(0.2)
            for item in connections:
                item.disconnect.set()
            # Each request ends quietly, with no response for a client that left.
            await asyncio.wait_for(asyncio.gather(*requests), LIGHT_READ_SECONDS)
            # The statements stop too, rather than running on detached.
            await _wait_until(lambda: holders.running == 0)
            assert interrupted.is_set()
            assert all(item.messages == [] for item in connections)
            # The same server's slots and connections are free for the next heavy read.
            transport = httpx.ASGITransport(app=server)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await asyncio.wait_for(
                    client.get("/api/tools/usage"), LIGHT_READ_SECONDS
                )
                assert response.status_code == 200, response.text
                summary = (await client.get("/api/diagnostics/requests")).json()
            (route,) = [item for item in summary["routes"] if item["route"] == "/api/tools/usage"]
            # A client that left is not a server failure: 499, counted with 4xx.
            assert (route["status_2xx"], route["status_4xx"], route["status_5xx"]) == (
                1,
                DEFAULT_HEAVY_READ_CONCURRENCY,
                0,
            )
        finally:
            holders.release.set()
            await store.close()

    asyncio.run(scenario())


def _limited_app(limiter: HeavyReadLimiter, gate: asyncio.Event, entered: list[str]) -> Any:
    api = FastAPI()

    @api.post("/heavy", dependencies=[limiter.dependency()])
    async def heavy(payload: Annotated[dict, Body()]) -> dict:
        entered.append(payload["name"])
        await gate.wait()
        return {"lane": current_read_lane().value}

    @api.get("/heavy", dependencies=[limiter.dependency()])
    async def heavy_get() -> dict:
        entered.append("get")
        await gate.wait()
        return {"lane": current_read_lane().value}

    api.add_middleware(ClientDisconnectMiddleware)
    return api


def test_a_client_that_stays_gets_its_heavy_response():
    async def scenario():
        gate, entered = asyncio.Event(), []
        api = _limited_app(HeavyReadLimiter(1), gate, entered)
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            posted = asyncio.create_task(client.post("/heavy", json={"name": "post"}))
            fetched = asyncio.create_task(client.get("/heavy"))
            await _wait_until(lambda: len(entered) == 1)
            await asyncio.sleep(0.05)
            assert len(entered) == 1
            gate.set()
            responses = await asyncio.gather(posted, fetched)
        assert [item.status_code for item in responses] == [200, 200]
        assert [item.json() for item in responses] == [{"lane": "bulk"}] * 2
        assert sorted(entered) == ["get", "post"]

    asyncio.run(scenario())


def test_a_disconnected_client_leaves_the_admission_queue_and_its_slot():
    async def scenario():
        gate, entered = asyncio.Event(), []
        limiter = HeavyReadLimiter(1)
        api = _limited_app(limiter, gate, entered)
        running = _Connection(b'{"name": "running"}')
        queued = _Connection(b'{"name": "queued"}')
        first = asyncio.create_task(
            api(
                _http_scope("POST", "/heavy", body=b'{"name": "running"}'),
                running.receive,
                running.send,
            )
        )
        await _wait_until(lambda: entered == ["running"])
        second = asyncio.create_task(
            api(
                _http_scope("POST", "/heavy", body=b'{"name": "queued"}'),
                queued.receive,
                queued.send,
            )
        )
        await _wait_until(lambda: limiter.waiting == 1)
        queued.disconnect.set()
        await asyncio.wait_for(second, LIGHT_READ_SECONDS)
        assert limiter.waiting == 0 and queued.messages == []
        running.disconnect.set()
        await asyncio.wait_for(first, LIGHT_READ_SECONDS)
        assert limiter.running == 0 and running.messages == []
        assert entered == ["running"]

    asyncio.run(scenario())
