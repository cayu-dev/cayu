"""Local host observation semantics; business owners are qualified separately."""

import asyncio

import pytest

from cayu.collaboration._host_lifecycle import HostLifecycle, HostPassOutcome


def test_inert_construction_and_real_repeated_cancellation():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def step(deadline, stop):
            calls.append((deadline, stop))
            entered.set()
            await release.wait()
            return HostPassOutcome(1, False)

        host = HostLifecycle(step, observation_timeout_s=10, poll_interval_s=0.01)
        assert calls == []
        assert not host.inspect().pending
        for _ in range(2):
            observer = asyncio.create_task(host.service_once())
            await entered.wait()
            await asyncio.sleep(0)
            observer.cancel()
            assert observer.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled()
            assert host.inspect().pending
        assert len(calls) == 1
        assert not calls[0][1].is_set()
        release.set()
        result = await host.service_once()
        assert result.outcome == HostPassOutcome(1, False)
        assert not result.pending
        await host.aclose(timeout_s=1)

    asyncio.run(scenario())


def test_timeout_and_close_retain_opaque_pass_without_redispatch():
    async def scenario():
        release = asyncio.Event()
        calls = []

        async def step(deadline, stop):
            calls.append(deadline)
            await release.wait()
            assert stop.is_set()
            return HostPassOutcome(0, False)

        host = HostLifecycle(step, observation_timeout_s=0.001, poll_interval_s=0.001)
        assert (await host.service_once()).pending
        assert (await host.service_once()).pending
        assert len(calls) == 1
        assert (await host.aclose(timeout_s=0.001)).pending
        assert host.closing
        with pytest.raises(RuntimeError, match="closing"):
            await host.service_once()
        release.set()
        assert not (await host.aclose(timeout_s=1)).pending

    asyncio.run(scenario())


def test_overlap_rejected_and_close_joins_servicing_not_callers_lifetime():
    async def scenario():
        entered = asyncio.Event()
        unrelated = asyncio.Event()

        async def step(deadline, stop):
            entered.set()
            await stop.wait()
            return HostPassOutcome(0, False)

        host = HostLifecycle(step, observation_timeout_s=10, poll_interval_s=0.01)

        async def caller():
            await host.service_once()
            await unrelated.wait()

        observer = asyncio.create_task(caller())
        await entered.wait()
        for method in (host.service_once, host.run):
            with pytest.raises(RuntimeError, match="already active"):
                await method()
        # The caller remains alive after service_once releases its ownership.
        async with asyncio.timeout(1):
            result = await host.aclose(timeout_s=10)
        assert not result.pending
        assert not observer.done()
        unrelated.set()
        await observer

    asyncio.run(scenario())


def test_cancelled_close_preserves_pass_and_original_error():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        failure = RuntimeError("native failure")

        async def step(deadline, stop):
            entered.set()
            await release.wait()
            raise failure

        host = HostLifecycle(step, observation_timeout_s=0.001, poll_interval_s=0.001)
        assert (await host.service_once()).pending
        close = asyncio.create_task(host.aclose(timeout_s=10))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already active"):
            await host.aclose(timeout_s=1)
        close.cancel()
        assert close.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await close
        assert close.cancelled()
        assert host.inspect().pending
        release.set()
        with pytest.raises(RuntimeError) as caught:
            await host.aclose(timeout_s=1)
        assert caught.value is failure

    asyncio.run(scenario())


def test_continuous_loop_closes_without_starting_another_pass():
    async def scenario():
        calls = []
        entered = asyncio.Event()

        async def step(deadline, stop):
            calls.append(deadline)
            entered.set()
            await stop.wait()
            return HostPassOutcome(1, True)

        host = HostLifecycle(step, observation_timeout_s=10, poll_interval_s=0.001)
        runner = asyncio.create_task(host.run())
        await entered.wait()
        await host.aclose(timeout_s=1)
        await runner
        assert len(calls) == 1

    asyncio.run(scenario())


def test_caller_cancellation_during_delivered_idle_timeout_is_not_duplicated(monkeypatch):
    async def scenario():
        async def step(deadline, stop):
            return HostPassOutcome(0, False)

        host = HostLifecycle(step, observation_timeout_s=1, poll_interval_s=0.01)
        runner = None
        counts = []
        original = asyncio.Timeout._on_timeout

        def deliver_timeout_then_cancel(timeout):
            # Let the actual idle deadline deliver its control signal first.
            original(timeout)
            if timeout._task is runner and not counts:
                counts.append(runner.cancelling())
                runner.cancel()
                counts.append(runner.cancelling())

        monkeypatch.setattr(asyncio.Timeout, "_on_timeout", deliver_timeout_then_cancel)
        runner = asyncio.create_task(host.run())
        async with asyncio.timeout(2):
            with pytest.raises(asyncio.CancelledError):
                await runner
        assert counts == [1, 2]
        assert runner.cancelled()
        # The timeout removes only its own signal; the caller signal survives.
        assert runner.cancelling() == 1
        assert not (await host.aclose(timeout_s=1)).pending

    asyncio.run(scenario())
