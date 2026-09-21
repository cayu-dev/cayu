from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import pytest
from tests.artifacts.test_resources import make_store, registered_resource, registered_transfer

from cayu._filesystem_lock import cooperative_path_lock
from cayu.artifacts import LocalArtifactStore
from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    MandateResourcePreparationReader,
    ResourceAcquisitionCommand,
    ResourceOwnerUnavailable,
    ResourceTransferCommand,
)
from cayu.collaboration._contracts import ExactMatch


def _hold_journal(root, ready, finish):
    with cooperative_path_lock(
        Path(root), "resource-owner", lock_directory_name="cayu-resource-owner-locks"
    ):
        ready.set()
        if not finish.wait(10):
            raise TimeoutError("journal test barrier expired")


def _reader_for_store(reader, store):
    return MandateResourcePreparationReader(
        owner=reader._owner,
        resolver=reader._resolver,
        context=reader._context,
        redactor=reader._redactor,
        registration=reader._registration,
        policy=reader._policy,
        artifact_store=store,
        collaboration_store=reader._collaboration_store,
        initialized=reader._initialized,
        responsibilities=tuple(
            pair
            for pair in reader._responsibilities
            if isinstance(pair[0], ResourceAcquisitionCommand)
        ),
        transfers=tuple(
            pair
            for pair in reader._responsibilities
            if isinstance(pair[0], ResourceTransferCommand)
        ),
    )


@pytest.mark.parametrize("failure", ["none", "unlock", "close", "both"])
@pytest.mark.parametrize("cancel", [False, True])
def test_aborted_journal_preserves_lock_cleanup_failures(tmp_path, monkeypatch, failure, cancel):
    fcntl = pytest.importorskip("fcntl")
    import os

    store, artifact = make_store(tmp_path)
    primary = ResourceOwnerUnavailable("preparation revoked")
    unlock_error = OSError("journal unlock acknowledgement lost")
    close_error = OSError("journal close acknowledgement lost")

    async def run():
        owner, cmd, permit, _, ledger, _, _ = await registered_resource(tmp_path, store, artifact)
        before = owner._journal.path.read_bytes()
        entered = asyncio.Event()
        finish = threading.Event()
        loop = asyncio.get_running_loop()
        original_flock, original_close = fcntl.flock, os.close
        armed = False
        closing = set()
        task = None

        @asynccontextmanager
        async def reject(command, lease):
            nonlocal armed
            armed = True
            raise primary
            yield  # pragma: no cover

        def flock(fd, operation):
            nonlocal armed
            if (
                armed
                and operation == fcntl.LOCK_UN
                and threading.current_thread().name == "cayu-resource-journal"
            ):
                armed = False
                closing.add(fd)
                loop.call_soon_threadsafe(entered.set)
                if not finish.wait(10):
                    raise TimeoutError("journal abort barrier expired")
                original_flock(fd, operation)
                if failure in {"unlock", "both"}:
                    raise unlock_error
                return
            return original_flock(fd, operation)

        def close(fd):
            original_close(fd)
            if fd in closing and threading.current_thread().name == "cayu-resource-journal":
                closing.remove(fd)
                if failure in {"close", "both"}:
                    raise close_error

        try:
            monkeypatch.setattr(owner._preparation_reader, "revalidation_guard", reject)
            monkeypatch.setattr(fcntl, "flock", flock)
            monkeypatch.setattr(os, "close", close)
            task = asyncio.create_task(owner.authorize(cmd, permit=permit))
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                task.cancel("first")
                task.cancel("second")
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail("caller cancellation was lost")
                assert task.cancelled() and task.cancelling() == 2
            finish.set()
            with pytest.raises(BaseException) as raised:
                await (owner.drain() if cancel else task)

            def leaves(error):
                assert type(error).__name__ != "Aborted"
                if isinstance(error, BaseExceptionGroup):
                    for child in error.exceptions:
                        yield from leaves(child)
                else:
                    yield error
                if error.__cause__ is not None:
                    yield from leaves(error.__cause__)
                elif not error.__suppress_context__ and error.__context__ is not None:
                    yield from leaves(error.__context__)

            expected: list[BaseException] = [primary]
            if failure in {"unlock", "both"}:
                expected.append(unlock_error)
            if failure in {"close", "both"}:
                expected.append(close_error)
            assert list(leaves(raised.value)) == expected
            assert owner._journal.path.read_bytes() == before
            assert not owner._workers
            await store.delete(artifact.id)
        finally:
            finish.set()
            if task is not None:
                with suppress(BaseException):
                    await task
            await ledger.close()

    asyncio.run(run())


def test_journal_commit_does_not_starve_resolver_executor(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)

    async def run():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        owner, cmd, permit, resolver, ledger, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        original = resolver.acquire

        @asynccontextmanager
        async def resolve(context):
            await asyncio.to_thread(lambda: None)
            async with original(context) as resolution:
                yield resolution

        monkeypatch.setattr(resolver, "acquire", resolve)
        try:
            prep = await owner.authorize(cmd, permit=permit)
            receipt = await owner.acquire(cmd, preparation=prep)
            await owner.release(receipt)
            await store.delete(artifact.id)
        finally:
            await ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("replacement_kind", ["different_root", "same_root"])
def test_cleanup_reopen_rejects_same_id_different_physical_store(
    tmp_path, monkeypatch, transfer, replacement_kind
):
    store, artifact = make_store(tmp_path)

    async def run():
        source, cmd, permit, _, ledger, _, _ = await registered_resource(tmp_path, store, artifact)
        destination_ledger = None
        try:
            prep = await source.authorize(cmd, permit=permit)
            source_receipt = await source.acquire(cmd, preparation=prep)
            target, receipt = source, source_receipt
            if transfer:
                target, cmd, _, _, destination_ledger, _, _ = await registered_transfer(
                    tmp_path / "destination", store, artifact, source, source_receipt
                )
                receipt = await target.accept_transfer(cmd, source_owner=source)

            def unexpected_write(value):
                raise AssertionError("Exact readback must not rewrite the journal")

            with monkeypatch.context() as patch:
                patch.setattr(target._journal, "_write", unexpected_write)
                lookup = await (target.read_transfer(cmd) if transfer else target.readback(cmd))
                assert isinstance(lookup, ExactMatch) and lookup.receipt == receipt

            async def fail(*args, **kwargs):
                raise OSError("unpin has not dispatched")

            with monkeypatch.context() as patch:
                patch.setattr(store, "_release_resource_pin", fail)
                with pytest.raises(OSError):
                    await (
                        target.release_transfer(receipt) if transfer else target.release(receipt)
                    )
            before = target._journal.path.read_bytes()
            replacement_root = tmp_path / "replacement"
            if replacement_kind == "same_root":
                store.root.rename(tmp_path / "original-artifacts")
                replacement_root = store.root
            replacement = LocalArtifactStore(replacement_root, store_id=store.id)
            with pytest.raises(ResourceOwnerUnavailable, match="identity conflicts"):
                LocalArtifactResourceOwner(
                    target._journal.root,
                    owner=target.owner,
                    artifact_store=replacement,
                    preparation_reader=_reader_for_store(target._preparation_reader, replacement),
                )
            assert target._journal.path.read_bytes() == before
            if replacement_kind == "same_root":
                with pytest.raises(ResourceOwnerUnavailable, match="physical identity"):
                    await target.reconcile()
                assert target._journal.path.read_bytes() == before
                replacement_root.rename(tmp_path / "unused-replacement")
                (tmp_path / "original-artifacts").rename(store.root)
            with pytest.raises(ValueError, match="durable pin"):
                await store.delete(artifact.id)
            reopened = LocalArtifactResourceOwner(
                target._journal.root,
                owner=target.owner,
                artifact_store=store,
                preparation_reader=target._preparation_reader,
            )
            target._preparation_reader._resolver.revoked = True
            await reopened.reconcile()
            family = "transfers" if transfer else "operations"
            state = json.loads(target._journal.path.read_bytes())[family]
            assert all(
                r["stage"] == "released" and r["responsibility_settled"] for r in state.values()
            )
            if transfer:
                await source.release(source_receipt)
            await store.delete(artifact.id)
        finally:
            await ledger.close()
            if destination_ledger is not None:
                await destination_ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("termination", ["timeout", "cancel"])
def test_stalled_journal_publication_remains_owned(
    tmp_path, monkeypatch, after_commit, termination
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    async def run():
        owner, cmd, permit, resolver, ledger, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        finish = threading.Event()
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()
        task = None
        try:
            prep = await owner.authorize(cmd, permit=permit)
            original = owner._journal._write

            def write(value):
                if any(r["stage"] == "owned" for r in value["operations"].values()):
                    if after_commit:
                        original(value)
                    loop.call_soon_threadsafe(entered.set)
                    if not finish.wait(10):
                        raise TimeoutError("publication test barrier expired")
                    if after_commit:
                        return
                original(value)

            monkeypatch.setattr(owner._journal, "_write", write)
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 3)
            task = asyncio.create_task(owner.acquire(cmd, preparation=prep))
            await asyncio.wait_for(entered.wait(), 5)
            # This runs while the journal worker is blocked. Revocation must not
            # be held behind slow commit settlement or undo dispatched work.
            resolver.revoked = True
            if termination == "cancel":
                task.cancel("first")
                task.cancel("second")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 2
            else:
                with pytest.raises(ResourceOwnerUnavailable, match="settlement remains owned"):
                    await task
            with pytest.raises(ResourceOwnerUnavailable, match="mutation remains owned"):
                await owner.reconcile()
            finish.set()
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 5)
            await owner.drain()
            state = json.loads(owner._journal.path.read_bytes())["operations"]
            receipt = next(iter(state.values()))["owned_receipt"]
            from cayu.artifacts.resources import ResourceAcquisitionReceipt

            retained = ResourceAcquisitionReceipt.model_validate(receipt)
            with pytest.raises(ValueError, match="durable pin"):
                await store.delete(artifact.id)
            resolver.revoked = False
            monkeypatch.setattr(owner._journal, "_write", original)
            await owner.release(retained)
            await store.delete(artifact.id)
        finally:
            finish.set()
            if task is not None:
                with suppress(BaseException):
                    await task
            await ledger.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "operation", ["acquire", "readback", "read_transfer", "release", "reconcile"]
)
@pytest.mark.parametrize("termination", ["timeout", "cancel"])
def test_public_journal_contention_keeps_loop_live_and_cleanup_owned(
    tmp_path, monkeypatch, operation, termination
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    async def run():
        source, cmd, permit, _, ledger, _, _ = await registered_resource(tmp_path, store, artifact)
        destination_ledger = None
        process = None
        finish = None
        heartbeat_task = None
        task = None
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        try:
            prep = await source.authorize(cmd, permit=permit)
            receipt = None
            target = source
            if operation != "acquire":
                receipt = await source.acquire(cmd, preparation=prep)
            if operation == "read_transfer":
                target, cmd, _, _, destination_ledger, _, _ = await registered_transfer(
                    tmp_path / "destination", store, artifact, source, receipt
                )
                transfer = await target.accept_transfer(cmd, source_owner=source)
            calls = {
                "acquire": lambda: target.acquire(cmd, preparation=prep),
                "readback": lambda: target.readback(cmd),
                "read_transfer": lambda: target.read_transfer(cmd),
                "release": lambda: target.release(receipt),
                "reconcile": lambda: target.reconcile(),
            }
            context = multiprocessing.get_context("spawn")
            ready, finish = context.Event(), context.Event()
            process = context.Process(
                target=_hold_journal, args=(str(target._journal.root), ready, finish)
            )
            process.start()
            assert await asyncio.to_thread(ready.wait, 5)
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 0.15)
            heartbeat_task = asyncio.create_task(heartbeat())
            started = time.monotonic()
            task = asyncio.create_task(calls[operation]())
            if termination == "cancel":
                await asyncio.sleep(0.03)
                task.cancel("first")
                task.cancel("second")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 2
            else:
                with pytest.raises(ResourceOwnerUnavailable, match="settlement remains owned"):
                    await task
            assert time.monotonic() - started < 2
            assert ticks >= 2
            with pytest.raises(ResourceOwnerUnavailable, match="mutation remains owned"):
                await target.reconcile()
            finish.set()
            await asyncio.to_thread(process.join, 5)
            assert process.exitcode == 0
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 5)
            with suppress(ResourceOwnerUnavailable):
                await target.drain()
            await target.reconcile()
            if operation == "read_transfer":
                await target.release_transfer(transfer)
            if receipt is not None:
                await source.release(receipt)
            await store.delete(artifact.id)
        finally:
            if finish is not None:
                finish.set()
            if task is not None and not task.done():
                with suppress(BaseException):
                    await task
            if process is not None:
                await asyncio.to_thread(process.join, 5)
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            await ledger.close()
            if destination_ledger is not None:
                await destination_ledger.close()

    asyncio.run(run())
