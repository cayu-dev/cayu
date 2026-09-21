from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import threading

import pytest
from tests.artifacts.test_resources import (
    LocalArtifactResourceOwner,
    authorized,
    command,
    make_store,
    owner,
    preparation_permit,
    registered_resource,
    registered_transfer,
)

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import ArtifactInputMember, ArtifactScope, FolderInputEntry, FolderInputManifest
from cayu.artifacts.resources import (
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
    ResourceOwnerUnsupported,
)
from cayu.collaboration._contracts import ExactUnavailable
from cayu.collaboration._preparation import ExactMatch
from cayu.vaults.redaction import SecretRedactor


def test_generic_release_cannot_remove_exact_resource_pins(tmp_path):
    store, artifact = make_store(tmp_path)

    async def run():
        source, cmd, permit, _, collaboration, _, _ = await registered_resource(
            tmp_path / "source", store, artifact
        )
        destination_store = None
        try:
            prep = await source.authorize(cmd, permit=permit)
            receipt = await source.acquire(cmd, preparation=prep)
            await store.release_pin(artifact.id, owner="resource:" + receipt.operation_digest)
            await store.release_pin(artifact.id, owner=source._pin_owner(receipt.operation_digest))
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            assert isinstance(await source.readback(cmd), ExactMatch)
            destination, transfer, _, _, destination_store, _, _ = await registered_transfer(
                tmp_path / "destination", store, artifact, source, receipt
            )
            accepted = await destination.accept_transfer(transfer, source_owner=source)
            await source.release_transferred_source(accepted, destination_owner=destination)
            await store.release_pin(artifact.id, owner=accepted.destination_pin_owner)
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            assert isinstance(await destination.read_transfer(transfer), ExactMatch)
            await destination.release_transfer(accepted)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("dimension", ["bytes", "nodes"])
@pytest.mark.parametrize("headroom", [-1, 0, 1])
def test_journal_reserves_cleanup_before_pin_dispatch(
    tmp_path, monkeypatch, transfer, dimension, headroom
):
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)

    def nodes(value):
        if isinstance(value, dict):
            return 1 + sum(nodes(item) for item in value.values())
        if isinstance(value, list):
            return 1 + sum(map(nodes, value))
        return 1

    async def run():
        (
            source,
            cmd,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(tmp_path / "source", store, artifact)
        destination_store = None
        try:
            preparation = await source.authorize(cmd, permit=permit)
            target = source
            receipt = None
            if transfer:
                receipt = await source.acquire(cmd, preparation=preparation)
                (
                    target,
                    cmd,
                    _,
                    resolver,
                    destination_store,
                    initialized,
                    participant,
                ) = await registered_transfer(
                    tmp_path / "destination", store, artifact, source, receipt
                )
            family = "transfers" if transfer else "operations"
            complete = "accepted" if transfer else "owned"
            original_write = target._journal._write
            original_pin = store._pin_resource
            armed = False
            failed = False
            pins = []

            async def pin(*args, **kwargs):
                await original_pin(*args, **kwargs)
                pins.append(args)

            def write(value):
                nonlocal armed, failed
                records = value[family]
                if not armed and records:
                    assert all(record["stage"] == "pending" for record in records.values())
                    envelope = resources._journal_capacity_envelope(value)
                    measure = (
                        (lambda item: len(canonical_durable_json_bytes(item, "test")))
                        if dimension == "bytes"
                        else nodes
                    )
                    limit = measure(envelope) + headroom
                    assert measure(value) < limit  # The pending write alone would fit.
                    monkeypatch.setattr(
                        resources, "RESOURCE_JOURNAL_MAX_" + dimension.upper(), limit
                    )
                    armed = True
                if not failed and any(record["stage"] == complete for record in records.values()):
                    assert pins  # Fail *after* the real external pin succeeded.
                    failed = True
                    raise OSError("post-pin publication failed")
                original_write(value)

            monkeypatch.setattr(store, "_pin_resource", pin)
            monkeypatch.setattr(target._journal, "_write", write)
            attempt = (
                target.accept_transfer(cmd, source_owner=source)
                if transfer
                else target.acquire(cmd, preparation=preparation)
            )
            if headroom < 0:
                with pytest.raises(ValueError):
                    await attempt
                assert not pins
                assert not target._journal._read()[family]
            else:
                with pytest.raises(OSError, match="post-pin"):
                    await attempt
                assert len(pins) == 1
                assert failed
                with pytest.raises(ValueError):
                    await store.delete(artifact.id)
                # Unrelated publication cannot steal the remaining cleanup budget.
                before = target._journal.path.read_bytes()
                member = ArtifactInputMember(
                    resource=target.artifact_selector(store, target.owner, artifact.id).resource,
                    content_sha256=hashlib.sha256(b"immutable").hexdigest(),
                    metadata_sha256=hashlib.sha256(
                        canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
                    ).hexdigest(),
                    size_bytes=9,
                )
                manifest = FolderInputManifest(
                    entries=tuple(
                        FolderInputEntry(
                            path=f"{index:02d}-" + "x" * 400, git_mode="100644", member=member
                        )
                        for index in range(32)
                    )
                )
                with pytest.raises(ValueError):
                    target.register_manifest(manifest)
                assert target._journal.path.read_bytes() == before
                resolver.revoked = True
                target = type(target)(
                    target._journal.root,
                    owner=target.owner,
                    artifact_store=store,
                    preparation_reader=target._preparation_reader,
                )
                await target.reconcile(source_owners=(source,) if transfer else ())
                await target.reconcile(source_owners=(source,) if transfer else ())
                record = next(iter(json.loads(target._journal.path.read_bytes())[family].values()))
                assert record["stage"] == "released"
                assert record["responsibility_settled"] is True
            ledger = destination_store if transfer else collaboration
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
            if receipt is not None:
                await source.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())


def test_same_operation_key_on_different_receivers_has_independent_pins(tmp_path):
    store, artifact = make_store(tmp_path)

    async def run():
        first = LocalArtifactResourceOwner(
            tmp_path / "first", owner=owner("first"), artifact_store=store
        )
        second = LocalArtifactResourceOwner(
            tmp_path / "second", owner=owner("second"), artifact_store=store
        )
        left = await authorized(first, command(first.owner, artifact.id, store=store))
        right = await authorized(second, command(second.owner, artifact.id, store=store))
        assert left.operation_digest == right.operation_digest
        await first.release(left)
        with pytest.raises(ValueError):
            await store.delete(artifact.id)
        assert isinstance(await second.readback(right.command), ExactMatch)
        await second.release(right)
        await store.delete(artifact.id)

    asyncio.run(run())


@pytest.mark.parametrize("bound", ["max_materials", "max_total_bytes", "max_manifest_bytes"])
def test_folder_bounds_refuse_without_retained_responsibility(tmp_path, bound):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import _reserved_bytes
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        members = [(artifact, b"immutable")]
        for index in range(2, 5):
            metadata = await store.put_bytes(
                b"member",
                artifact_id="art_" + str(index) * 32,
                filename="member",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            members.append((metadata, b"member"))
        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        (
            resource,
            cmd,
            permit,
            _,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(
            tmp_path,
            store,
            artifact,
            collaboration,
            folder_members=tuple(members),
            bounds={bound: 1024 if bound == "max_manifest_bytes" else 1},
        )
        try:
            for _ in range(2):
                with pytest.raises(ResourceOwnerUnsupported, match="bounds"):
                    await resource.authorize(cmd, permit=permit)
                resource = type(resource)(
                    tmp_path / "mandated-owner",
                    owner=resource.owner,
                    artifact_store=store,
                    preparation_reader=resource._preparation_reader,
                )
                await resource.reconcile()
                with resource._journal.locked() as journal:
                    assert not journal["operations"]
                    assert not journal["authorizations"]
                    assert not journal["events"]
                    assert _reserved_bytes(journal) == 0
                _, obligations = await collaboration.scan_obligations(
                    initialized,
                    participant,
                    after=0,
                    limit=64,
                    pending_only=True,
                    retention_revision=None,
                    redactor=SecretRedactor(),
                )
                assert not obligations
            for metadata, _ in members:
                await store.delete(metadata.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "changed", [{"revision": 2}, {"incarnation": "invented"}, {"incarnation": "0" * 64}]
)
def test_fresh_artifact_identity_must_match_publication(tmp_path, changed):
    store, artifact = make_store(tmp_path)
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=owner("source"), artifact_store=store
    )
    cmd = command(resource.owner, artifact.id, store=store)
    assert resource.artifact_selector(store, resource.owner, artifact.id) == cmd.intent.selector
    wrong = cmd.model_copy(
        update={
            "intent": cmd.intent.model_copy(
                update={
                    "selector": cmd.intent.selector.model_copy(
                        update={"resource": cmd.intent.selector.resource.model_copy(update=changed)}
                    )
                }
            )
        }
    )
    with pytest.raises(ResourceOwnerConflict):
        asyncio.run(resource.authorize(wrong, permit=preparation_permit(wrong)))
    with resource._journal.locked() as journal:
        assert not journal["operations"]
        assert not journal["authorizations"]
    receipt = asyncio.run(authorized(resource, cmd))
    asyncio.run(resource.release(receipt))


@pytest.mark.parametrize("changed", [{"revision": 2}, {"incarnation": "invented"}])
def test_folder_member_identity_is_not_caller_chosen(tmp_path, changed):
    store, artifact = make_store(tmp_path)
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=owner("source"), artifact_store=store
    )
    member = ArtifactInputMember(
        resource=resource.artifact_selector(store, resource.owner, artifact.id).resource.model_copy(
            update=changed
        ),
        content_sha256=hashlib.sha256(b"immutable").hexdigest(),
        metadata_sha256=hashlib.sha256(
            canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
        ).hexdigest(),
        size_bytes=9,
    )
    with pytest.raises(ResourceOwnerConflict):
        resource.register_manifest(
            FolderInputManifest(
                entries=(FolderInputEntry(path="input", git_mode="100644", member=member),)
            )
        )
    with resource._journal.locked() as journal:
        assert not journal["manifests"]
    asyncio.run(store.delete(artifact.id))


def test_closure_deletion_honors_exact_pin_namespace(tmp_path):
    from cayu.artifacts import LocalArtifactStore

    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"session", filename="input", session_id="session")
        resource = LocalArtifactResourceOwner(
            tmp_path / "owner", owner=owner("source"), artifact_store=store
        )
        receipt = await authorized(resource, command(resource.owner, artifact.id, store=store))
        claim = await store.claim_session_closure(
            "session", "a" * 64, max_records=10, max_bytes=10000
        )
        await store.release_pin(artifact.id, owner="resource:" + receipt.operation_digest)
        with pytest.raises(ValueError, match="pin"):
            await store.delete_session_closure_artifact(claim, artifact.id)
        await resource.release(receipt)
        await store.delete_session_closure_artifact(claim, artifact.id)

    asyncio.run(run())


def test_republished_artifact_does_not_satisfy_old_selector(tmp_path):
    store, artifact = make_store(tmp_path)
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=owner("source"), artifact_store=store
    )
    cmd = command(resource.owner, artifact.id, store=store)
    asyncio.run(store.delete(artifact.id))
    asyncio.run(
        store.put_bytes(
            b"replacement",
            artifact_id=artifact.id,
            filename="replacement",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    with pytest.raises(ResourceOwnerConflict):
        asyncio.run(authorized(resource, cmd))
    new = command(resource.owner, artifact.id, store=store)
    assert new.intent.selector != cmd.intent.selector
    receipt = asyncio.run(authorized(resource, new))
    asyncio.run(resource.release(receipt))


def test_artifact_replacement_between_authorization_and_pin_is_not_accepted(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, cmd, permit, _, collaboration, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        original = store._pin_resource

        async def replace_then_pin(artifact_id, *, owner, dispatch):
            await store.delete(artifact_id)
            await store.put_bytes(
                b"replaced",
                artifact_id=artifact_id,
                filename="replacement",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            await original(artifact_id, owner=owner, dispatch=dispatch)

        try:
            prep = await resource.authorize(cmd, permit=permit)
            monkeypatch.setattr(store, "_pin_resource", replace_then_pin)
            from cayu.artifacts.resources import ResourceOwnerUnavailable

            with pytest.raises(ResourceOwnerUnavailable):
                await resource.acquire(cmd, preparation=prep)
            assert isinstance(await resource.readback(cmd), ExactUnavailable)
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            monkeypatch.setattr(store, "_pin_resource", original)
            await resource.reconcile()
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_retained_impossible_folder_is_cleaned_on_recovery(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import _reserved_bytes
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        second = await store.put_bytes(
            b"second",
            artifact_id="art_" + "2" * 32,
            filename="second",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        (
            resource,
            cmd,
            permit,
            _,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(
            tmp_path,
            store,
            artifact,
            collaboration,
            folder_members=((artifact, b"immutable"), (second, b"second")),
            bounds={"max_total_bytes": 10},
        )
        try:
            # Fault injection bypasses only deterministic preflight, exercising
            # permanent rejection after real registration and both pin dispatches.
            monkeypatch.setattr(resource, "_validate_bounds", lambda command: None)
            prep = await resource.authorize(cmd, permit=permit)
            with pytest.raises(ResourceOwnerUnsupported, match="total bytes"):
                await resource.acquire(cmd, preparation=prep)
            for metadata in (artifact, second):
                with pytest.raises(ValueError):
                    await store.delete(metadata.id)
            reopened = type(resource)(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=resource._preparation_reader,
            )
            await reopened.reconcile()
            await reopened.reconcile()
            with reopened._journal.locked() as journal:
                assert _reserved_bytes(journal) == 0
                assert all(
                    record["stage"] == "released" for record in journal["operations"].values()
                )
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            for metadata in (artifact, second):
                await store.delete(metadata.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["uncertain", "releasing", "released"])
def test_transfer_readback_requires_active_acceptance(tmp_path, monkeypatch, stage):
    store, artifact = make_store(tmp_path)

    async def run():
        source = LocalArtifactResourceOwner(
            tmp_path / "source", owner=owner("source"), artifact_store=store
        )
        receipt = await authorized(source, command(source.owner, artifact.id, store=store))
        destination, transfer, _, _, collaboration, _, _ = await registered_transfer(
            tmp_path / "destination", store, artifact, source, receipt
        )
        try:
            if stage == "uncertain":
                original = store._pin_resource

                async def lose_ack(*args, **kwargs):
                    await original(*args, **kwargs)
                    raise OSError("lost pin acknowledgement")

                monkeypatch.setattr(store, "_pin_resource", lose_ack)
                with pytest.raises(OSError):
                    await destination.accept_transfer(transfer, source_owner=source)
                monkeypatch.setattr(store, "_pin_resource", original)
            else:
                accepted = await destination.accept_transfer(transfer, source_owner=source)
                assert isinstance(await destination.read_transfer(transfer), ExactMatch)
                if stage == "releasing":
                    original = store._release_resource_pin

                    async def fail_release(*args, **kwargs):
                        raise OSError("release blocked")

                    monkeypatch.setattr(store, "_release_resource_pin", fail_release)
                    with pytest.raises(OSError):
                        await destination.release_transfer(accepted)
                    monkeypatch.setattr(store, "_release_resource_pin", original)
                else:
                    await destination.release_transfer(accepted)
            assert isinstance(await destination.read_transfer(transfer), ExactUnavailable)
            if stage == "uncertain":
                await destination.reconcile(source_owners=(source,))
                accepted = (await destination.read_transfer(transfer)).receipt
                await destination.release_transfer(accepted)
            elif stage == "releasing":
                await destination.release_transfer(accepted)
            await source.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("recovery", ["public", "reconcile"])
def test_release_finishes_after_unpinned_material_disappears(
    tmp_path, monkeypatch, replacement, recovery
):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import _reserved_bytes
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        resource, cmd, permit, resolver, _, initialized, participant = await registered_resource(
            tmp_path, store, artifact, collaboration
        )
        original = store._release_resource_pin

        async def lose_ack(*args, **kwargs):
            await original(*args, **kwargs)
            raise OSError("unpin committed but acknowledgement lost")

        try:
            receipt = await resource.acquire(
                cmd, preparation=await resource.authorize(cmd, permit=permit)
            )
            monkeypatch.setattr(store, "_release_resource_pin", lose_ack)
            with pytest.raises(OSError, match="acknowledgement"):
                await resource.release(receipt)
            monkeypatch.setattr(store, "_release_resource_pin", original)
            await store.delete(artifact.id)
            if replacement:
                await store.put_bytes(
                    b"replacement",
                    artifact_id=artifact.id,
                    filename="new",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="resource",
                )
                await store.pin(artifact.id, owner="replacement-owner")
            reopened = type(resource)(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=resource._preparation_reader,
            )
            if recovery == "public":
                # Receipt possession still does not bypass current cleanup authority.
                from cayu.collaboration.mandates import MandateDenied

                resolver.revoked = True
                with pytest.raises(MandateDenied):
                    await reopened.release(receipt)
                resolver.revoked = False
                await reopened.release(receipt)
                await reopened.release(receipt)
            else:
                resolver.revoked = True
                await reopened.reconcile()
                await reopened.reconcile()
            with reopened._journal.locked() as journal:
                record = journal["operations"][receipt.operation_digest]
                assert record["stage"] == "released"
                assert record["responsibility_settled"]
                assert _reserved_bytes(journal) == 0
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            if replacement:
                assert (await store.read_bytes(artifact.id)).content == b"replacement"
                with pytest.raises(ValueError):
                    await store.delete(artifact.id)
                await store.release_pin(artifact.id, owner="replacement-owner")
                await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("entrance", ["authorize", "acquire"])
def test_metadata_lock_wait_is_bounded_and_owned(tmp_path, monkeypatch, cancel, entrance):
    import cayu.artifacts.local as local
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)
    locked, finish, observing = threading.Event(), threading.Event(), threading.Event()

    async def run():
        resource, cmd, permit, _, collaboration, _, _ = await registered_resource(
            tmp_path, store, artifact
        )
        prep = await resource.authorize(cmd, permit=permit)
        original_sync = local._sync_descriptor
        original_metadata = store._resource_metadata
        original_timeout = resources.RESOURCE_FOREGROUND_TIMEOUT_S

        def blocked_sync(descriptor):
            original_sync(descriptor)
            if not locked.is_set():
                locked.set()
                if not finish.wait(10):
                    raise AssertionError("locked filesystem worker was not released")

        def metadata(artifact_id):
            observing.set()
            return original_metadata(artifact_id)

        monkeypatch.setattr(local, "_sync_descriptor", blocked_sync)
        monkeypatch.setattr(store, "_resource_metadata", metadata)
        monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 0.1)
        holder = asyncio.create_task(store.pin(artifact.id, owner="competing"))
        task = None

        def attempt():
            if entrance == "authorize":
                return resource.authorize(cmd, permit=permit)
            return resource.acquire(cmd, preparation=prep)

        try:
            async with asyncio.timeout(3):
                while not locked.is_set():
                    await asyncio.sleep(0.001)
                task = asyncio.create_task(attempt())
                while not observing.is_set():
                    await asyncio.sleep(0.001)
                if cancel:
                    task.cancel("caller cancelled")
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled()
                    assert task.cancelling() == 1
                else:
                    with pytest.raises(ResourceOwnerUnavailable):
                        await task
                    assert not task.cancelled()
                assert not finish.is_set()
                assert resource._workers
                with pytest.raises(ResourceOwnerUnavailable):
                    await attempt()
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout)
            finish.set()
            await holder
            with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                await resource.drain()
            if entrance == "authorize":
                await resource.authorize(cmd, permit=permit)
            await resource.acquire(cmd)
            lookup = await resource.readback(cmd)
            assert isinstance(lookup, ExactMatch)
            await resource.release(lookup.receipt)
            await store.release_pin(artifact.id, owner="competing")
            await store.delete(artifact.id)
        finally:
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout)
            finish.set()
            await holder
            if task is not None and not task.done():
                await task
            await resource.drain()
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "operations",
    [
        tuple(values)
        for size in range(1, 4)
        for values in itertools.combinations(("read", "release", "transfer"), size)
    ],
)
def test_every_accepted_operation_set_can_settle(tmp_path, operations):
    store, artifact = make_store(tmp_path)

    async def run():
        if "release" not in operations:
            from pydantic import ValidationError

            from cayu.artifacts.resources import ResourceAcquisitionIntent

            raw = command(owner("source"), artifact.id, store=store).intent.model_dump(mode="json")
            raw["allowed_operations"] = operations
            with pytest.raises(ValidationError, match="mandatory release"):
                ResourceAcquisitionIntent.model_validate(raw)
            await store.delete(artifact.id)
            return
        (
            resource,
            cmd,
            permit,
            _,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(
            tmp_path, store, artifact, bounds={"allowed_operations": operations}
        )
        try:
            receipt = await resource.acquire(
                cmd, preparation=await resource.authorize(cmd, permit=permit)
            )
            await resource.release(receipt)
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("lost_ack", [False, True])
def test_accepted_handoff_recovery_does_not_need_source_release_grant(
    tmp_path, monkeypatch, lost_ack
):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.collaboration.mandates import MandateDenied
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        source_path = tmp_path / "source"
        source_path.mkdir()
        source_store = SQLiteCollaborationStore(source_path / "collaboration.sqlite")
        source, cmd, permit, resolver, _, initialized, participant = await registered_resource(
            source_path, store, artifact, source_store
        )
        destination_store = None
        try:
            receipt = await source.acquire(
                cmd, preparation=await source.authorize(cmd, permit=permit)
            )
            destination, transfer, _, _, destination_store, _, _ = await registered_transfer(
                tmp_path / "destination", store, artifact, source, receipt
            )
            accepted = await destination.accept_transfer(transfer, source_owner=source)
            resolver.revoked = True
            with pytest.raises(MandateDenied):
                await source.release_transferred_source(accepted, destination_owner=destination)
            # Recovery starts from independently reopened durable owner state.
            source = type(source)(
                source_path / "mandated-owner",
                owner=source.owner,
                artifact_store=store,
                preparation_reader=source._preparation_reader,
            )
            destination = type(destination)(
                tmp_path / "destination" / "mandated-owner",
                owner=destination.owner,
                artifact_store=store,
                preparation_reader=destination._preparation_reader,
            )
            if lost_ack:
                original = source._set_record

                async def commit_then_raise(digest, stage, value, **kwargs):
                    await original(digest, stage, value, **kwargs)
                    if stage == "released":
                        raise OSError("source cleanup acknowledgement lost")

                monkeypatch.setattr(source, "_set_record", commit_then_raise)
                with pytest.raises(OSError, match="acknowledgement"):
                    await destination.reconcile(source_owners=(source,))
                monkeypatch.setattr(source, "_set_record", original)
            await destination.reconcile(source_owners=(source,))
            await destination.reconcile(source_owners=(source,))
            _, obligations = await source_store.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert not obligations
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            assert isinstance(await destination.read_transfer(transfer), ExactMatch)
            await destination.release_transfer(accepted)
            await store.delete(artifact.id)
        finally:
            await source_store.close()
            if destination_store is not None:
                await destination_store.close()

    asyncio.run(run())
