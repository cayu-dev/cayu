"""Local read ownership never turns a cancelled observer into a new source call."""

import asyncio

import pytest

from cayu.collaboration._host_reads import HostReads


def test_cancelled_read_observer_preserves_exact_query_capacity_and_original_failure():
    async def scenario():
        reads = HostReads(slots=2, bytes_limit=65536)
        entered, release = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("read cleanup failure")
        calls = 0

        async def blocked():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            raise failure

        observer = asyncio.create_task(
            reads.observe("blocked", expectation=b"exact", reserved_bytes=32768, read=blocked)
        )
        await entered.wait()
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        assert reads.pending == 1
        with pytest.raises(ValueError, match="conflicts"):
            await reads.observe(
                "blocked", expectation=b"changed", reserved_bytes=32768, read=blocked
            )

        async def good():
            return "independent"

        assert (
            await reads.observe(
                "too-large", expectation=b"different", reserved_bytes=65536, read=good
            )
            is None
        )
        result = await reads.observe("good", expectation=b"exact", reserved_bytes=32768, read=good)
        assert result.value == "independent"
        assert reads.pending == 1
        assert (
            await reads.observe("blocked", expectation=b"exact", reserved_bytes=32768, read=blocked)
            is None
        )
        assert calls == 1
        assert await reads.close(0.001) == ()
        assert reads.pending == 1
        release.set()
        errors = await reads.close(1)
        assert errors == (failure,)
        assert reads.pending == 0
        assert await reads.close(0) == ()
        assert (
            await reads.observe("closed", expectation=b"exact", reserved_bytes=32768, read=good)
            is None
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("through_close", [False, True])
def test_read_cancelled_before_entry_preserves_signal_and_releases_only_read_capacity(
    through_close,
):
    async def scenario():
        reads = HostReads(slots=2, bytes_limit=65536)
        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        cancelled_tasks = []
        calls = 0

        async def read():
            nonlocal calls
            calls += 1

        def cancel_before_entry(loop, coroutine, **kwargs):
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
            task.cancel()
            task.cancel()
            cancelled_tasks.append(task)
            return task

        loop.set_task_factory(cancel_before_entry)
        try:
            if through_close:
                # Start the observer separately, but install the factory only
                # for the read task it creates. The observer is not cancelled.
                observer = asyncio.Task(
                    reads.observe("read", expectation=b"exact", reserved_bytes=32768, read=read),
                    loop=loop,
                )
                await asyncio.sleep(0)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert reads.pending == 1
            else:
                with pytest.raises(asyncio.CancelledError):
                    await reads.observe(
                        "read", expectation=b"exact", reserved_bytes=32768, read=read
                    )
                assert reads.pending == 0
        finally:
            loop.set_task_factory(previous_factory)
        assert calls == 0
        assert len(cancelled_tasks) == 1
        assert cancelled_tasks[0].cancelled()
        assert cancelled_tasks[0].cancelling() == 2
        errors = await reads.close(1)
        if through_close:
            assert len(errors) == 1 and isinstance(errors[0], asyncio.CancelledError)
        else:
            assert errors == ()
        assert reads.pending == 0
        assert await reads.close(0) == ()

    asyncio.run(scenario())
