from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, suppress

import pytest
from tests.artifacts.test_resources import make_store, registered_resource, registered_transfer

from cayu.artifacts import ArtifactScope
from cayu.artifacts.resources import ResourceOwnerUnavailable
from cayu.collaboration.mandates import MandateDenied
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("kind", ["artifact", "folder", "transfer"])
@pytest.mark.parametrize(
    "window",
    [
        "positive",
        "before_dispatch",
        "in_flight",
        "publication",
        "guard_exit_failure",
        "guard_exit_cancel",
        "guard_exit_and_pin_failure",
    ],
)
def test_public_resource_revocation_is_atomic_with_dispatch_and_publication(
    tmp_path, monkeypatch, kind, window
):
    import cayu.artifacts.local as local

    store, artifact = make_store(tmp_path)

    async def run():
        members = ()
        if kind in {"folder", "transfer"}:
            second = await store.put_bytes(
                b"second",
                filename="second.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            members = ((artifact, b"immutable"), (second, b"second"))
        (
            source,
            command,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(tmp_path / "source", store, artifact, folder_members=members)
        target = source
        source_receipt = None
        destination_store = None
        release_thread = threading.Event()
        proceed = asyncio.Event()
        task = None
        try:
            prep = await source.authorize(command, permit=permit)
            if kind == "transfer":
                source_receipt = await source.acquire(command, preparation=prep)
                (
                    target,
                    command,
                    _,
                    resolver,
                    destination_store,
                    initialized,
                    participant,
                ) = await registered_transfer(
                    tmp_path / "destination", store, artifact, source, source_receipt
                )
            lock = asyncio.Lock()
            entered = asyncio.Event()
            loop = asyncio.get_running_loop()
            dispatches = []
            published = []
            guard_failure = OSError("guard exit failed")
            pin_failure = OSError("pin acknowledgement lost")
            exit_failure = window.startswith("guard_exit")
            original_acquire = resolver.acquire
            original_pin = store._pin_resource
            original_change = local._change_resource_pin
            original_executor = loop.run_in_executor
            original_commit = target._journal._dispatch_commit
            family = "transfers" if kind == "transfer" else "operations"
            terminal = "accepted" if kind == "transfer" else "owned"

            @asynccontextmanager
            async def guarded(context):
                async with lock, original_acquire(context) as resolution:
                    yield resolution
                # A resolver may suspend while exiting its guard. Publication
                # must precede this exit, rather than following revalidate().
                if window == "publication" and published and not entered.is_set():
                    entered.set()
                    await proceed.wait()
                if exit_failure and dispatches:
                    entered.set()
                    raise guard_failure

            async def revoke():
                async with lock:
                    resolver.revoked = True

            async def pin(*args, **kwargs):
                if window == "before_dispatch" and not entered.is_set():
                    entered.set()
                    await proceed.wait()
                return await original_pin(*args, **kwargs)

            def change(*args):
                if args[-1] and (window == "in_flight" or exit_failure):
                    loop.call_soon_threadsafe(entered.set)
                    if not release_thread.wait(15):
                        raise TimeoutError("test dispatch barrier timed out")
                result = original_change(*args)
                if args[-1] and window == "guard_exit_and_pin_failure":
                    raise pin_failure
                return result

            def executor(executor, func, *args):
                if args and args[0] is change and args[-1] is True:
                    # Submission, not coroutine scheduling, is the irreversible
                    # boundary. Revocation must still be serialized here.
                    assert lock.locked()
                    assert not resolver.revoked
                    dispatches.append(args)
                return original_executor(executor, func, *args)

            def commit(value, finish):
                if any(record["stage"] == terminal for record in value[family].values()):
                    assert lock.locked()
                    assert not resolver.revoked
                    published.append(terminal)
                return original_commit(value, finish)

            monkeypatch.setattr(resolver, "acquire", guarded)
            monkeypatch.setattr(store, "_pin_resource", pin)
            monkeypatch.setattr(local, "_change_resource_pin", change)
            monkeypatch.setattr(loop, "run_in_executor", executor)
            monkeypatch.setattr(target._journal, "_dispatch_commit", commit)
            task = asyncio.create_task(
                target.accept_transfer(command, source_owner=source)
                if kind == "transfer"
                else target.acquire(command, preparation=prep)
            )
            async with asyncio.timeout(20):
                if window != "positive":
                    await entered.wait()
                    # Revocation finishes even while the submitted filesystem
                    # worker remains blocked: no resolver lock spans settlement.
                    await asyncio.wait_for(revoke(), 1)
                    if window == "in_flight" or exit_failure:
                        assert not task.done()
                        with pytest.raises(ResourceOwnerUnavailable):
                            await target.reconcile()
                    if window == "guard_exit_cancel":
                        task.cancel("first")
                        task.cancel("second")
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        assert task.cancelled() and task.cancelling() == 2
                        with pytest.raises(ResourceOwnerUnavailable):
                            await target.reconcile()
                    proceed.set()
                    release_thread.set()
                if exit_failure:
                    with pytest.raises((OSError, ExceptionGroup)) as raised:
                        if window == "guard_exit_cancel":
                            await target.drain()
                        else:
                            await task
                    if window == "guard_exit_and_pin_failure":
                        assert raised.value.exceptions == (guard_failure, pin_failure)
                    else:
                        assert raised.value is guard_failure
                    assert len(dispatches) == 1 and not published
                elif window in {"before_dispatch", "in_flight"}:
                    with pytest.raises(MandateDenied):
                        await task
                    assert len(dispatches) == (0 if window == "before_dispatch" else 1)
                    assert not published
                else:
                    receipt = await task
                    assert receipt.stage == terminal
                    assert len(dispatches) == (1 if kind == "artifact" else 2)
                    assert len(published) == 1
            monkeypatch.setattr(target._journal, "_dispatch_commit", original_commit)
            monkeypatch.setattr(loop, "run_in_executor", original_executor)
            if window in {"before_dispatch", "in_flight"} or exit_failure:
                await target.reconcile()
            else:
                resolver.revoked = False
                if kind == "transfer":
                    await target.release_transfer(receipt)
                else:
                    await target.release(receipt)
            ledger = destination_store or collaboration
            _, obligations = await ledger.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            if source_receipt is not None:
                await source.release(source_receipt)
            await store.delete(artifact.id)
            if members:
                await store.delete(second.id)
        finally:
            proceed.set()
            release_thread.set()
            if task is not None and not task.done():
                with suppress(BaseException):
                    await task
            await collaboration.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())
