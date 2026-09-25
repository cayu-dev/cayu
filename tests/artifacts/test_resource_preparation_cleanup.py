"""Exact native cleanup survives pre-dispatch gaps, revocation and lost ACKs."""

import asyncio
import time

import pytest
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.artifacts.test_resources import (
    _RegisteredTestPreparationReader,
    make_store,
    registered_resource,
)
from tests.core.test_participant_identity import stores as stores

from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
    ResourceOwnerUnsupported,
    resource_operation_digest,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def artifact_store(tmp_path):
    return make_store(tmp_path)


@pytest.mark.parametrize("phase", ["authorization", "dispatch"])
async def test_planned_deadline_bounds_native_lease_and_late_pin(
    tmp_path, artifact_store, monkeypatch, phase
):
    store, artifact = artifact_store
    deadline = int(time.time() * 1000) + 60_000
    owner, command, permit, _, ledger, _, _ = await registered_resource(
        tmp_path, store, artifact, bounds={"deadline_at_ms": deadline}
    )
    reader = owner._preparation_reader
    pinned = []
    pin = store._pin_resource

    async def counted(*args, **kwargs):
        pinned.append(args)
        return await pin(*args, **kwargs)

    monkeypatch.setattr(store, "_pin_resource", counted)
    try:
        if phase == "authorization":
            monkeypatch.setattr("cayu.artifacts.resources.time.time", lambda: deadline / 1000)
            with pytest.raises(ResourceOwnerUnavailable):
                await owner.authorize(command, permit=permit)
        else:
            prepared = await owner.authorize(command, permit=permit)
            assert prepared.lease.expires_at_ms == deadline
            register = reader.register_responsibility

            async def expires_after_registration(*args):
                await register(*args)
                monkeypatch.setattr("cayu.artifacts.resources.time.time", lambda: deadline / 1000)

            monkeypatch.setattr(reader, "register_responsibility", expires_after_registration)
            with pytest.raises(ResourceOwnerUnavailable):
                await owner.acquire(command, preparation=prepared)
        assert pinned == []
        settled = await owner.settle_preparation(command, permit=permit)
        assert settled.proves_exclusion
        with pytest.raises(ResourceOwnerConflict):
            await owner.settle_preparation(
                command.model_copy(
                    update={
                        "intent": command.intent.model_copy(update={"deadline_at_ms": deadline + 1})
                    }
                ),
                permit=permit,
            )
        await store.delete(artifact.id)
    finally:
        await owner.drain()
        await ledger.close()


@pytest.mark.parametrize("damage", ["release_event", "authorization_commitment"])
async def test_settlement_readback_rejects_divergent_native_evidence(
    tmp_path, artifact_store, damage
):
    store, artifact = artifact_store
    owner, command, permit, _, ledger, _, _ = await registered_resource(tmp_path, store, artifact)
    try:
        await owner.authorize(command, permit=permit)
        await owner.settle_preparation(command, permit=permit)
        digest = resource_operation_digest(command)
        async with owner._journal.transaction() as journal:
            if damage == "release_event":
                journal["events"] = [
                    item
                    for item in journal["events"]
                    if not (item.get("operation") == digest and item.get("stage") == "released")
                ]
            else:
                journal["authorizations"][digest]["receipt"]["command_bytes_sha256"] = "0" * 64
        with pytest.raises(ResourceOwnerUnavailable):
            await owner.settle_preparation(command, permit=permit)
    finally:
        await owner.drain()
        await ledger.close()


async def test_unqualified_cleanup_reader_refuses_without_native_mutation(tmp_path, artifact_store):
    store, artifact = artifact_store
    original, command, permit, _, ledger, _, _ = await registered_resource(
        tmp_path, store, artifact
    )
    owner = LocalArtifactResourceOwner(
        tmp_path / "unqualified",
        owner=original.owner,
        artifact_store=store,
        preparation_reader=_RegisteredTestPreparationReader(original.owner),
    )
    before = owner._journal.path.read_bytes()
    try:
        with pytest.raises(ResourceOwnerUnsupported, match="cleanup is not qualified"):
            await owner.settle_preparation(command, permit=permit)
        assert owner._journal.path.read_bytes() == before
    finally:
        await original.drain()
        await owner.drain()
        await ledger.close()


async def test_cancelled_cleanup_observer_keeps_late_acquisition_fenced(
    tmp_path, artifact_store, monkeypatch
):
    store, artifact = artifact_store
    owner, command, permit, _, ledger, _, _ = await registered_resource(tmp_path, store, artifact)
    prepared = await owner.authorize(command, permit=permit)
    reader = owner._preparation_reader
    reached, release = asyncio.Event(), asyncio.Event()
    settle = reader.settle_responsibility

    async def paused(*args):
        reached.set()
        await release.wait()
        return await settle(*args)

    monkeypatch.setattr(reader, "settle_responsibility", paused)
    other = LocalArtifactResourceOwner(
        owner._journal.root, owner=owner.owner, artifact_store=store, preparation_reader=reader
    )
    observer = asyncio.create_task(owner.settle_preparation(command, permit=permit))
    competing = None
    try:
        async with asyncio.timeout(20):
            await reached.wait()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 1
        competing = asyncio.create_task(other.acquire(command, preparation=prepared))
        await asyncio.sleep(0)
        assert not competing.done()
        release.set()
        await owner.drain()
        with pytest.raises(ResourceOwnerUnavailable):
            await competing
        settled = await other.settle_preparation(command, permit=permit)
        assert settled.proves_exclusion
        await store.delete(artifact.id)
    finally:
        release.set()
        await asyncio.gather(
            observer, *(() if competing is None else (competing,)), return_exceptions=True
        )
        await owner.drain()
        await other.drain()
        await ledger.close()


@pytest.mark.parametrize("accepted", [False, True])
async def test_transfer_cleanup_fences_late_acceptance(tmp_path, artifact_store, accepted):
    store, artifact = artifact_store
    (
        source,
        command,
        permit,
        destination,
        template,
        reader,
        _,
        source_ledger,
        ledger,
    ) = await registered_template(tmp_path, store, artifact)
    try:
        acquired = await source.acquire(
            command, preparation=await source.authorize(command, permit=permit)
        )
        transfer = template.bind(acquired)
        if accepted:
            await destination.accept_transfer(transfer, source_owner=source)
        receiving_permit = reader.transfer_permit(transfer)
        before = destination._journal.path.read_bytes()
        forged = template.bind(acquired.model_copy(update={"receipt_id": "untrusted-receipt"}))
        with pytest.raises(ResourceOwnerConflict):
            await destination.settle_preparation(
                forged, permit=receiving_permit, source_owner=source
            )
        assert destination._journal.path.read_bytes() == before
        source._preparation_reader._resolver.revoked = True
        settled = await destination.settle_preparation(
            transfer, permit=receiving_permit, source_owner=source
        )
        assert settled.expected == receiving_permit and settled.proves_exclusion
        assert await destination.settle_preparation(transfer, permit=receiving_permit) == settled
        source._preparation_reader._resolver.revoked = False
        with pytest.raises(ResourceOwnerUnavailable):
            await destination.accept_transfer(transfer, source_owner=source)
        await source.release(acquired)
        await store.delete(artifact.id)
    finally:
        await source.drain()
        await destination.drain()
        await source_ledger.close()
        await ledger.close()


@pytest.mark.parametrize("phase", ["absent", "prepared", "owned"])
async def test_exact_cleanup_without_renewing_acquisition_authority(
    stores, tmp_path, artifact_store, monkeypatch, phase
):
    store, artifact = artifact_store
    owner, command, permit, resolver, ledger, _, _ = await registered_resource(
        tmp_path, store, artifact, collaboration_store=stores()
    )
    reader = owner._preparation_reader
    active_resolution = resolver.resolution
    try:
        if phase != "absent":
            prepared = await owner.authorize(command, permit=permit)
            if phase == "owned":
                await owner.acquire(command, preparation=prepared)
        # Remove acquisition rights but retain separately granted cleanup.
        resolver.resolution = active_resolution.model_copy(
            update={
                "principal": active_resolution.principal.model_copy(
                    update={"actions": ("release",)}
                ),
                "chain": active_resolution.chain.model_copy(
                    update={
                        "entries": tuple(
                            entry.model_copy(update={"actions": ("release",)})
                            for entry in active_resolution.chain.entries
                        )
                    }
                ),
            }
        )
        settle = reader.settle_responsibility

        async def lost_ack(*args):
            await settle(*args)
            raise OSError("settlement acknowledgement lost")

        monkeypatch.setattr(reader, "settle_responsibility", lost_ack)
        with pytest.raises(OSError, match="acknowledgement lost"):
            await owner.settle_preparation(command, permit=permit)
        await owner.drain()
        monkeypatch.setattr(reader, "settle_responsibility", settle)
        owner = LocalArtifactResourceOwner(
            owner._journal.root,
            owner=owner.owner,
            artifact_store=store,
            preparation_reader=reader,
        )
        receipt = await owner.settle_preparation(command, permit=permit)
        assert receipt.expected == permit and receipt.proves_exclusion
        assert await owner.settle_preparation(command, permit=permit) == receipt
        before = owner._journal.path.read_bytes()
        with pytest.raises(ResourceOwnerConflict):
            await owner.settle_preparation(
                command.model_copy(
                    update={
                        "intent": command.intent.model_copy(
                            update={"max_total_bytes": command.intent.max_total_bytes + 1}
                        )
                    }
                ),
                permit=permit,
            )
        assert owner._journal.path.read_bytes() == before
        resolver.resolution = active_resolution
        prepared = await owner.authorize(command, permit=permit)
        with pytest.raises(ResourceOwnerUnavailable):
            await owner.acquire(command, preparation=prepared)
        # Positive physical cleanup evidence, not only a terminal status.
        await store.delete(artifact.id)
    finally:
        await owner.drain()
        await ledger.close()
