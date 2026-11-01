"""Bounded concurrent reads and shared session loads for overview readers."""

from __future__ import annotations

import asyncio

import pytest

from cayu.sessions.overview import SessionOverviewReader, gather_in_order


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
        names = ("slow", "fails") if failure else ("slow",)
        task = asyncio.create_task(
            gather_in_order([lambda name=name: reader._load(name) for name in names])
        )
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
