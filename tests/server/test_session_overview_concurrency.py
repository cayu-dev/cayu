"""Overview pages read their sessions concurrently, bounded, and in page order."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _start

from cayu.server import ServerConfig, create_server
from cayu.sessions.overview import (
    OVERVIEW_READ_CONCURRENCY,
    SessionOverviewReader,
    gather_in_order,
)

__all__ = ["conformance_postgres_dsn", "session_store_case"]


def test_gather_in_order_bounds_concurrency_and_keeps_call_order():
    async def scenario():
        running = peak = 0

        async def call(index: int) -> int:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            # Later calls finish first, so completion order differs from call order.
            await asyncio.sleep(0.001 * (10 - index))
            running -= 1
            return index

        result = await gather_in_order([lambda i=i: call(i) for i in range(10)], limit=3)
        assert result == list(range(10))
        assert peak == 3

    asyncio.run(scenario())


def test_gather_in_order_raises_the_first_failure_unwrapped_and_cancels_the_rest():
    async def scenario():
        cancelled: list[int] = []
        started = asyncio.Event()

        async def slow(index: int) -> int:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(index)
                raise
            return index

        async def failing() -> int:
            await started.wait()
            raise KeyError("missing-session")

        calls = [lambda: slow(0), failing, lambda: slow(2)]
        with pytest.raises(KeyError, match="missing-session"):
            await gather_in_order(calls, limit=3)
        assert sorted(cancelled) == [0, 2]

    asyncio.run(scenario())


def test_concurrent_loads_of_one_session_share_one_store_read():
    class CountingStore:
        def __init__(self) -> None:
            self.loads = 0

        async def load(self, session_id: str):
            self.loads += 1
            await asyncio.sleep(0.01)
            return None

    async def scenario():
        store = CountingStore()
        reader = SessionOverviewReader(store)  # type: ignore[arg-type]
        results = await asyncio.gather(*(reader._load("parent") for _ in range(5)))
        assert results == [None] * 5
        assert store.loads == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_overview_page_reads_sessions_concurrently_in_page_order(session_store_case, monkeypatch):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            ids = [f"session-{index}" for index in range(8)]
            for session_id in ids:
                await _start(app, session_id, "writer")
            native = store.summarize_outcome
            running = peak = 0

            async def observed(session_id, *args, **kwargs):
                nonlocal running, peak
                running += 1
                peak = max(peak, running)
                try:
                    await asyncio.sleep(0.02)
                    return await native(session_id, *args, **kwargs)
                finally:
                    running -= 1

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                expected = await client.get("/api/sessions", params={"include": "overview"})
                assert expected.status_code == 200, expected.text
                monkeypatch.setattr(store, "summarize_outcome", observed)
                response = await client.get("/api/sessions", params={"include": "overview"})
            assert response.status_code == 200, response.text
            body = response.json()
            assert sorted(row["id"] for row in body["sessions"]) == sorted(ids)
            # Same rows, same order, same overviews as a read without the delay.
            assert [row["id"] for row in body["sessions"]] == [
                row["id"] for row in expected.json()["sessions"]
            ]
            assert [row["session_id"] for row in body["overviews"]] == [
                row["id"] for row in body["sessions"]
            ]
            assert body["overviews"] == expected.json()["overviews"]
            assert 1 < peak <= OVERVIEW_READ_CONCURRENCY
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True], ids=["cancel", "store_failure"])
def test_abandoned_page_drains_underlying_reads(failure):
    async def scenario():
        started = asyncio.Event()
        drained = asyncio.Event()

        class BlockingStore:
            async def load(self, session_id):
                if session_id == "fails":
                    await started.wait()
                    raise KeyError("missing")
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    drained.set()

        reader = SessionOverviewReader(BlockingStore())
        task = asyncio.create_task(reader.read(("slow", "fails") if failure else ("slow",)))
        await started.wait()
        if not failure:
            task.cancel()
        with pytest.raises(KeyError if failure else asyncio.CancelledError):
            await task
        assert drained.is_set()
        assert not reader._loading
        assert not reader._load_waiters

    asyncio.run(scenario())


def test_cancelling_one_shared_reader_preserves_the_other():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        loads = 0

        class BlockingStore:
            async def load(self, session_id):
                nonlocal loads
                loads += 1
                started.set()
                await release.wait()
                return None

        reader = SessionOverviewReader(BlockingStore())
        first = asyncio.create_task(reader._load("parent"))
        second = asyncio.create_task(reader._load("parent"))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second is None
        assert loads == 1
        assert not reader._loading
        assert not reader._load_waiters

    asyncio.run(scenario())
