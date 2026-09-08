"""Shutdown owns channel cleanup, not the ambient server caller task."""

import asyncio

import pytest

from cayu.runtime._browser_control_channels import BrowserGuestChannels
from cayu.runtime.browser_control import BrowserControlConflict


def test_shutdown_retains_live_cleanup_until_positive_completion():
    async def scenario():
        owner = BrowserGuestChannels()
        entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def serve(settlement):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await release.wait()
                settlement.confirm()

        caller = asyncio.create_task(owner.run(serve))
        await entered.wait()
        assert not await owner.drain(timeout_s=0.01)
        assert cleanup.is_set() and not caller.done()
        assert caller.cancelling() == 0
        with pytest.raises(BrowserControlConflict):
            await owner.run(serve)
        release.set()
        await caller
        assert await owner.drain(timeout_s=0.01)
        assert not owner._tasks

    asyncio.run(scenario())


def test_cancellation_with_failed_cleanup_does_not_claim_settlement():
    async def scenario():
        owner = BrowserGuestChannels()
        entered = asyncio.Event()
        failure = OSError("fence publication failed")

        async def serve(settlement):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from failure

        caller = asyncio.create_task(owner.run(serve))
        await entered.wait()
        assert not await owner.drain(timeout_s=0.1)
        with pytest.raises(asyncio.CancelledError) as error:
            await caller
        assert caller.cancelled()
        assert error.value.__cause__ is failure
        assert not await owner.drain(timeout_s=0.01)
        assert len(owner._tasks) == 1

    asyncio.run(scenario())


def test_confirmed_cleanup_preserves_cancellation_without_retaining_historical_cause():
    async def scenario():
        owner = BrowserGuestChannels()
        entered = asyncio.Event()
        historical = RuntimeError("earlier protocol failure")

        async def serve(settlement):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                settlement.confirm()
                raise cancellation from historical

        caller = asyncio.create_task(owner.run(serve))
        await entered.wait()
        assert caller.cancel("operator channel cancelled")
        with pytest.raises(asyncio.CancelledError) as raised:
            await caller
        assert caller.cancelled() and caller.cancelling() == 1
        assert raised.value.__cause__ is historical
        assert not owner._tasks and not owner._settlements and not owner._failed_cancellations
        assert await owner.drain()

    asyncio.run(scenario())
