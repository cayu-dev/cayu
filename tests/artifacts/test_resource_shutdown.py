from __future__ import annotations

import asyncio
import threading
from contextlib import suppress

import pytest
from tests.artifacts.test_resources import make_store, registered_resource, registered_transfer

from cayu.artifacts import ArtifactScope
from cayu.artifacts.resources import LocalArtifactResourceOwner, ResourceOwnerUnavailable
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("operation", ["artifact", "folder", "transfer", "release"])
def test_runner_shutdown_retains_dispatched_resource_work(tmp_path, monkeypatch, operation):
    import cayu.artifacts.local as local
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)
    dispatched, finish, shutting_down = threading.Event(), threading.Event(), threading.Event()
    armed, cleanup_started = threading.Event(), threading.Event()
    state = {}
    errors = []
    original_change = local._change_resource_pin
    original_cancel = asyncio.runners._cancel_all_tasks  # ty: ignore[unresolved-attribute]

    def change(*args):
        if armed.is_set():
            acquire = args[-1]
            if acquire == (operation != "release") and not dispatched.is_set():
                # Real executor dispatch, before the physical artifact lock.
                dispatched.set()
                if not finish.wait(15):
                    raise TimeoutError("shutdown pin barrier expired")
            elif not acquire:
                cleanup_started.set()
        return original_change(*args)

    def cancel_all(loop):
        # Observe the real Runner shutdown after its cancellation deliveries.
        loop.call_later(0.05, shutting_down.set)
        return original_cancel(loop)

    monkeypatch.setattr(local, "_change_resource_pin", change)
    monkeypatch.setattr(asyncio.runners, "_cancel_all_tasks", cancel_all)

    async def main():
        members = ()
        if operation == "folder":
            second = await store.put_bytes(
                b"second",
                filename="second.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            members = ((artifact, b"immutable"), (second, b"second"))
        source, cmd, permit, _, ledger, initialized, participant = await registered_resource(
            tmp_path / "source", store, artifact, folder_members=members
        )
        prep = await source.authorize(cmd, permit=permit)
        target = source
        state.update(
            source=source, ledgers=[ledger], materials=[artifact, *(m for m, _ in members[1:])]
        )
        if operation in {"transfer", "release"}:
            receipt = await source.acquire(cmd, preparation=prep)
            state["source_receipt"] = receipt
        if operation == "transfer":
            target, cmd, _, _, ledger, initialized, participant = await registered_transfer(
                tmp_path / "destination", store, artifact, source, receipt
            )
            state["ledgers"].append(ledger)
        state.update(target=target, ledger=ledger, initialized=initialized, participant=participant)
        armed.set()
        if operation == "transfer":
            call = target.accept_transfer(cmd, source_owner=source)
        elif operation == "release":
            call = source.release(receipt)
        else:
            call = source.acquire(cmd, preparation=prep)
        task = asyncio.create_task(call)
        async with asyncio.timeout(10):
            while not dispatched.is_set():
                await asyncio.sleep(0.005)
        task.cancel("first")
        task.cancel("second")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 2
        # Deliberately return without drain: Runner must cancel retained tasks.

    def run():
        try:
            asyncio.run(main())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, name="resource-shutdown-test")
    thread.start()
    try:
        assert dispatched.wait(10), errors
        assert shutting_down.wait(5)
        assert thread.is_alive()
        if operation != "release":
            assert not cleanup_started.wait(0.15), "cleanup overtook the dispatched pin"
        target = state["target"]
        competitor = LocalArtifactResourceOwner(
            target._journal.root,
            owner=target.owner,
            artifact_store=store,
            preparation_reader=target._preparation_reader,
        )
        with pytest.raises(ResourceOwnerUnavailable, match="mutation remains owned"):
            asyncio.run(competitor.reconcile())
    finally:
        finish.set()
        thread.join(15)
    assert not thread.is_alive()
    assert not errors
    monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 5)

    async def cleanup():
        target = state["target"]
        try:
            with suppress(BaseException):
                await target.drain()
            target._preparation_reader._resolver.revoked = True
            await target.reconcile()
            if operation == "transfer":
                await state["source"].release(state["source_receipt"])
            _, obligations = await state["ledger"].scan_obligations(
                state["initialized"],
                state["participant"],
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            for metadata in state["materials"]:
                await store.delete(metadata.id)
        finally:
            for ledger in state["ledgers"]:
                await ledger.close()

    asyncio.run(cleanup())
