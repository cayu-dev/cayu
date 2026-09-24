"""Internal dependency observation retains the same mutation owner."""

import asyncio

import pytest

from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
@pytest.mark.parametrize("termination", ["complete", "cancel", "deadline"])
async def test_owned_dependency_waits_without_releasing_mutation(termination):
    owners = _MutationOwners()
    owners.observation_timeout = 0.001
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def mutation():
        calls.append(True)
        started.set()
        await release.wait()
        return "committed"

    async def observe(*, wait):
        return await owners.run(
            mutation,
            key=("exact-operation",),
            expectation=b"complete-intent",
            redactor=SecretRedactor(),
            wait_for_settlement=wait,
        )

    timer = asyncio.timeout(None)

    async def runtime():
        async with timer:
            return await observe(wait=True)

    task = asyncio.create_task(runtime())
    try:
        await asyncio.wait_for(started.wait(), 5)
        # The public sibling times out while the runtime still owns its wait.
        with pytest.raises(CollaborationUnavailable, match="acknowledgement is pending"):
            await observe(wait=False)
        assert not task.done()
        assert len(owners.pending) == 1
        if termination == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            assert len(owners.pending) == 1
        elif termination == "deadline":
            timer.reschedule(asyncio.get_running_loop().time())
            with pytest.raises(TimeoutError):
                await task
            assert timer.expired()
            assert not task.cancelled() and task.cancelling() == 0
            assert len(owners.pending) == 1
        mutation_task = next(iter(owners.pending))
        release.set()
        assert await mutation_task == "committed"
        if termination == "complete":
            assert await task == "committed"
            assert not task.cancelled() and task.cancelling() == 0
        await asyncio.sleep(0)
        assert not owners.pending
        assert calls == [True]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owners.drain()
