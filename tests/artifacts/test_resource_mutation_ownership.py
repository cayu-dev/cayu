from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager, suppress

import pytest
from tests.artifacts.test_resources import make_store, registered_resource

from cayu.artifacts.resources import ResourceOwnerUnavailable


@pytest.mark.parametrize("operation", ["acquire", "release"])
@pytest.mark.parametrize("phase", ["before", "held", "exit"])
@pytest.mark.parametrize("termination", ["cancel", "timeout"])
def test_mutation_fence_io_is_bounded_and_late_entry_cannot_dispatch(
    tmp_path, monkeypatch, operation, phase, termination
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    async def run():
        owner, command, permit, _, ledger, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        finish = threading.Event()
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()
        task = None
        heartbeat = None
        calls = []
        ticks = 0
        armed = True
        try:
            preparation = await owner.authorize(command, permit=permit)
            receipt = None
            if operation == "release":
                receipt = await owner.acquire(command, preparation=preparation)
            original_lock = resources.cooperative_path_lock
            original_pin = store._pin_resource
            original_unpin = store._release_resource_pin

            def pause():
                loop.call_soon_threadsafe(entered.set)
                if not finish.wait(12):
                    raise TimeoutError("mutation lock test barrier expired")

            @contextmanager
            def lock(root, name, **kwargs):
                nonlocal armed
                intercept = name == "mutation" and root == owner._journal.root and armed
                if intercept:
                    armed = False
                if intercept and phase == "before":
                    pause()
                with original_lock(root, name, **kwargs):
                    if intercept and phase == "held":
                        pause()
                    try:
                        yield
                    finally:
                        if intercept and phase == "exit":
                            pause()

            async def pin(*args, **kwargs):
                calls.append("pin")
                return await original_pin(*args, **kwargs)

            async def unpin(*args, **kwargs):
                calls.append("unpin")
                return await original_unpin(*args, **kwargs)

            async def tick():
                nonlocal ticks
                while True:
                    ticks += 1
                    await asyncio.sleep(0.01)

            monkeypatch.setattr(resources, "cooperative_path_lock", lock)
            monkeypatch.setattr(store, "_pin_resource", pin)
            monkeypatch.setattr(store, "_release_resource_pin", unpin)
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 3)
            heartbeat = asyncio.create_task(tick())
            task = asyncio.create_task(
                owner.acquire(command, preparation=preparation)
                if operation == "acquire"
                else owner.release(receipt)
            )
            await asyncio.wait_for(entered.wait(), 5)
            count = ticks
            await asyncio.sleep(0.03)
            assert ticks > count
            started = time.monotonic()
            if termination == "cancel":
                task.cancel("first")
                task.cancel("second")
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail("caller cancellation was swallowed")
                assert task.cancelled() and task.cancelling() == 2
            else:
                with pytest.raises(ResourceOwnerUnavailable, match="settlement remains owned"):
                    await task
            assert time.monotonic() - started < 5
            if phase == "before":
                # No fence has been acquired yet. A competing owner can win;
                # the old invocation must not perform work after it later wins.
                await owner.reconcile()
            else:
                with pytest.raises(ResourceOwnerUnavailable, match="mutation remains owned"):
                    await owner.reconcile()
            if phase != "exit":
                assert not calls
            finish.set()
            if phase == "exit":
                await owner.drain()
                assert calls == (["pin"] if operation == "acquire" else ["unpin"])
            else:
                with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                    await owner.drain()
                assert not calls
            monkeypatch.setattr(resources, "cooperative_path_lock", original_lock)
            if operation == "acquire" and phase == "exit":
                receipt = (await owner.readback(command)).receipt
            if receipt is not None:
                await owner.release(receipt)
            await store.delete(artifact.id)
        finally:
            finish.set()
            if task is not None:
                with suppress(BaseException):
                    await task
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            await ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_mutation_fence_preserves_primary_and_unlock_failure(tmp_path, monkeypatch, cancel):
    fcntl = pytest.importorskip("fcntl")
    store, artifact = make_store(tmp_path)
    primary = OSError("resource read failed")
    cleanup = OSError("unlock acknowledgement lost")

    async def run():
        owner, command, permit, resolver, ledger, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        finish = threading.Event()
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()
        task = None
        armed = True
        try:
            prep = await owner.authorize(command, permit=permit)
            original_flock = fcntl.flock
            original_read = store.read_bytes

            async def read(*args, **kwargs):
                await original_read(*args, **kwargs)
                raise primary

            def flock(fd, operation):
                nonlocal armed
                if (
                    armed
                    and operation == fcntl.LOCK_UN
                    and threading.current_thread().name == "cayu-resource-mutation"
                ):
                    armed = False
                    loop.call_soon_threadsafe(entered.set)
                    if not finish.wait(10):
                        raise TimeoutError("unlock test barrier expired")
                    original_flock(fd, operation)
                    raise cleanup
                return original_flock(fd, operation)

            monkeypatch.setattr(store, "read_bytes", read)
            monkeypatch.setattr(fcntl, "flock", flock)
            task = asyncio.create_task(owner.acquire(command, preparation=prep))
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                task.cancel("first")
                task.cancel("second")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 2
            with pytest.raises(ResourceOwnerUnavailable, match="mutation remains owned"):
                await owner.reconcile()
            finish.set()
            with pytest.raises(OSError) as raised:
                await (owner.drain() if cancel else task)
            assert raised.value is primary
            assert raised.value.__cause__ is cleanup
            assert cleanup.__context__ is None
            monkeypatch.setattr(store, "read_bytes", original_read)
            monkeypatch.setattr(fcntl, "flock", original_flock)
            resolver.revoked = True
            await owner.reconcile()
            await store.delete(artifact.id)
        finally:
            finish.set()
            if task is not None:
                with suppress(BaseException):
                    await task
            await ledger.close()

    asyncio.run(run())
