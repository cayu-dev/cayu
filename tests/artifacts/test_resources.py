from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import (
    ArtifactInputMember,
    ArtifactScope,
    FolderInputEntry,
    FolderInputManifest,
    LocalArtifactStore,
)
from cayu.artifacts.aws_s3 import S3ArtifactStore
from cayu.artifacts.resources import (
    LocalArtifactResourceOwner as _LocalArtifactResourceOwner,
)
from cayu.artifacts.resources import (
    ResourceAcquisitionCommand,
    ResourceAcquisitionIntent,
    ResourceAcquisitionReceipt,
    ResourceOwnerConflict,
    ResourceOwnerError,
    ResourceOwnerUnavailable,
    ResourceOwnerUnsupported,
    ResourcePreparationAuthorization,
    ResourcePreparationLease,
    ResourcePreparationReader,
    ResourceTransferCommand,
    ResourceTransferIntent,
    resource_operation_digest,
)
from cayu.collaboration._contracts import (
    ExactUnavailable,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._permits import PermitCommand, PermitIntent, PermitRegistration
from cayu.collaboration._preparation import ExactMatch
from cayu.collaboration.mandates import (
    CollaborationMandate,
    MandateAccessContext,
    MandateChain,
    MandateResolution,
    MandateResolver,
    MandateRestrictions,
    PrincipalResolution,
    ResourceSelector,
)
from cayu.collaboration.participants import CollaborationLimits, ParticipantRef
from cayu.vaults.redaction import SecretRedactor


class _RegisteredTestPreparationReader(ResourcePreparationReader):
    def __init__(self, owner_ref, participant=None):
        self._owner = owner_ref
        self._participant = participant

    @property
    def owner(self):
        return self._owner

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def acquire(self, command, permit):
        yield ResourcePreparationAuthorization(
            command=command,
            permit=permit,
            lease=ResourcePreparationLease(
                operation_digest=resource_operation_digest(command),
                receiver=self.owner,
                resolver_generation=0,
                receiver_generation=0,
                revocation_generation=0,
                expires_at_ms=4_102_444_800_000,
                nonce="test-registration-lease",
                authority_sha256="0" * 64,
                authority_json="{}",
                chain_generations=(),
            ),
        )

    @asynccontextmanager
    async def revalidation_guard(self, command, lease):
        if lease.receiver != self.owner or lease.operation_digest != resource_operation_digest(
            command
        ):
            raise ResourceOwnerConflict("test lease mismatch")
        yield

    async def authorize_release(self, command):
        return

    async def register_responsibility(self, command, permit):
        return

    def transfer_permit(self, command):
        permit = preparation_permit(command)
        if self._participant is None:
            return permit
        return permit.model_copy(
            update={
                "intent": permit.intent.model_copy(
                    update={
                        "request": permit.intent.request.model_copy(
                            update={"participant": self._participant}
                        )
                    }
                )
            }
        )

    async def settle_responsibility(self, command, permit, reader):
        return


class _UnavailablePreparationReader(_RegisteredTestPreparationReader):
    @asynccontextmanager
    async def revalidation_guard(self, command, lease):
        raise ResourceOwnerUnavailable("authority unavailable")
        yield  # pragma: no cover - this reader always refuses


class LocalArtifactResourceOwner(_LocalArtifactResourceOwner):
    def __init__(self, root, *, owner, artifact_store, preparation_reader=None, participant=None):
        super().__init__(
            root,
            owner=owner,
            artifact_store=artifact_store,
            preparation_reader=preparation_reader
            or _RegisteredTestPreparationReader(owner, participant),
        )


class _MandateFixtureResolver(MandateResolver):
    def __init__(self, resolution):
        self.resolution = resolution
        self.revoked = False

    @property
    def ref(self):
        return self.resolution.principal.resolver

    @asynccontextmanager
    async def acquire(self, context):
        if self.revoked:
            from cayu.collaboration.mandates import MandateDenied

            raise MandateDenied()
        yield self.resolution


def mandate_reader(resource, command, *, expires_at_ms=4_102_444_800_000):
    resolver_ref = ObjectRef(
        owner=resource.owner,
        kind="mandate_resolver",
        object_id="resolver",
        incarnation="one",
        revision=1,
    )
    mandate_ref = ObjectRef(
        owner=resource.owner,
        kind="mandate",
        object_id="mandate",
        incarnation="one",
        revision=1,
    )
    selector = command.intent.selector
    restrictions = MandateRestrictions(
        channels=("artifact",),
        excluded_sources=(),
        independence_policy=mandate_ref,
        disclosure_policy=mandate_ref,
    )
    root = CollaborationMandate(
        reference=mandate_ref,
        root=mandate_ref,
        parent=None,
        issuer=resource.owner,
        principal="principal",
        participant=None,
        audiences=(resource.owner,),
        scopes=(resource.owner.application_scope,),
        actions=("prepare",),
        resources=(selector,),
        remaining_delegations=1,
        sponsor=None,
        budgets=(),
        restrictions=restrictions,
        expires_at_ms=expires_at_ms,
        revocation_generation=1,
    )
    resolution = MandateResolution(
        principal=PrincipalResolution(
            resolver=resolver_ref,
            issuer=resource.owner,
            principal="principal",
            participants=(),
            audiences=(resource.owner,),
            scopes=(resource.owner.application_scope,),
            actions=("prepare",),
            expires_at_ms=expires_at_ms,
        ),
        chain=MandateChain(entries=(root,)),
    )
    return _MandateFixtureResolver(resolution), MandateAccessContext(
        issuer=resource.owner,
        principal="principal",
        mandate=mandate_ref,
    )


class BlockingReadStore(LocalArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.started = asyncio.Event()
        self.release_read = asyncio.Event()

    async def read_bytes(self, artifact_id, *, max_bytes=None):
        self.started.set()
        await self.release_read.wait()
        return await super().read_bytes(artifact_id, max_bytes=max_bytes)


class FailingReadStore(LocalArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.read_count = 0
        self.fail_reads = False

    async def read_bytes(self, artifact_id, *, max_bytes=None):
        self.read_count += 1
        if self.fail_reads and self.read_count == 2:
            raise OSError("simulated second-member loss")
        return await super().read_bytes(artifact_id, max_bytes=max_bytes)


class CrashAfterPinStore(LocalArtifactStore):
    async def _pin_resource(self, artifact_id, *, owner, dispatch):
        await super()._pin_resource(artifact_id, owner=owner, dispatch=dispatch)
        os._exit(0)


class CrashAfterReleaseStore(LocalArtifactStore):
    async def _release_resource_pin(self, artifact_id, *, owner):
        await super()._release_resource_pin(artifact_id, owner=owner)
        os._exit(0)


class FailReleaseStore(LocalArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.fail_once = True

    async def _release_resource_pin(self, artifact_id, *, owner):
        if self.fail_once:
            self.fail_once = False
            raise OSError("simulated release interruption")
        return await super()._release_resource_pin(artifact_id, owner=owner)


def _crash_after_pending(owner_root: str, store_root: str, owner_ref: OwnerRef, artifact_id: str):
    store = CrashAfterPinStore(store_root)
    resource = LocalArtifactResourceOwner(owner_root, owner=owner_ref, artifact_store=store)
    cmd = command(owner_ref, artifact_id, store=store)
    asyncio.run(authorized(resource, cmd))


def _crash_transfer(
    source_root: str,
    destination_root: str,
    store_root: str,
    source_ref: OwnerRef,
    destination_ref: OwnerRef,
    transfer: ResourceTransferCommand,
):
    store = CrashAfterPinStore(store_root)
    source = LocalArtifactResourceOwner(source_root, owner=source_ref, artifact_store=store)
    destination = LocalArtifactResourceOwner(
        destination_root, owner=destination_ref, artifact_store=store
    )
    asyncio.run(destination.accept_transfer(transfer, source_owner=source))


def _crash_release(owner_root: str, store_root: str, owner_ref: OwnerRef, receipt):
    store = CrashAfterReleaseStore(store_root)
    resource = LocalArtifactResourceOwner(owner_root, owner=owner_ref, artifact_store=store)
    asyncio.run(resource.release(receipt))


def _delete_from_process(store_root: str, artifact_id: str, result):
    store = LocalArtifactStore(store_root)
    try:
        asyncio.run(store.delete(artifact_id))
    except BaseException as error:
        result.put(type(error).__name__)
    else:
        result.put("deleted")


def owner(name: str) -> OwnerRef:
    return OwnerRef(application_scope="app", owner_id=name, incarnation="one")


async def authorized(resource, cmd):
    preparation = await resource.authorize(cmd, permit=preparation_permit(cmd))
    return await resource.acquire(cmd, preparation=preparation)


def preparation_permit(cmd):
    from cayu.artifacts.resources import _preparation_target

    receiving_owner = cmd.destination
    request = PermitRegistration(
        operation=cmd.operation.model_copy(
            update={"caller_key": cmd.operation.caller_key + "-permit"}
        ),
        participant=ParticipantRef(
            owner=receiving_owner, participant_id="resource", incarnation="one"
        ),
        expected_lifecycle_revision=1,
        admission_generation=1,
        source_operation=cmd.operation.model_copy(update={"caller_key": cmd.operation.caller_key}),
        target=_preparation_target(cmd),
        target_state="future",
        effect_scope="transfer" if isinstance(cmd, ResourceTransferCommand) else "acquire",
        required_settlement="exclusion",
        settlement_operation=cmd.operation.model_copy(
            update={"caller_key": cmd.operation.caller_key + "-settle"}
        ),
    )
    return PermitCommand(
        operation=request.operation,
        source=receiving_owner,
        destination=receiving_owner,
        initiator=cmd.initiator.model_copy(update={"issuer": receiving_owner}),
        intent=PermitIntent(
            request=request,
            limits=CollaborationLimits(
                participants=2,
                aliases=2,
                operations=8,
                events=8,
                retained_bytes=4096,
                control_operations=1,
                control_events=1,
                control_bytes=1,
                namespaces=2,
                generations=2,
                obligations=2,
            ),
        ),
    )


def command(
    resource_owner: OwnerRef, artifact_id: str, *, key="fixed", store=None
) -> ResourceAcquisitionCommand:
    selector = ResourceSelector(
        resource=ObjectRef(
            owner=resource_owner,
            kind="artifact",
            object_id=artifact_id,
            incarnation="unresolved",
            revision=1,
        )
    )
    if store is not None:
        selector = _LocalArtifactResourceOwner.artifact_selector(store, resource_owner, artifact_id)
    intent = ResourceAcquisitionIntent(
        selector=selector,
        allowed_operations=("read", "transfer", "release"),
        max_materials=1,
        max_manifest_bytes=4096,
        max_total_bytes=1024 * 1024,
        cleanup_owner=resource_owner,
        policy=ObjectRef(
            owner=resource_owner,
            kind="resource_policy",
            object_id="local",
            incarnation="one",
            revision=1,
        ),
    )
    return ResourceAcquisitionCommand(
        operation=OperationRef(
            application_scope=resource_owner.application_scope,
            namespace_incarnation="namespace",
            generation=1,
            caller_key=key,
        ),
        source=resource_owner,
        destination=resource_owner,
        initiator=InitiatorBinding(
            issuer=resource_owner,
            principal="principal",
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        ),
        intent=intent,
    )


def make_store(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"immutable",
            artifact_id="art_" + "1" * 32,
            filename="input.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    return store, artifact


def test_exact_acquire_readback_replay_and_release(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_owner = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_owner, artifact_store=store
    )
    cmd = command(resource_owner, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, cmd))
    assert receipt.stage == "owned"
    looked_up = asyncio.run(resource.readback(cmd))
    assert isinstance(looked_up, ExactMatch)
    assert looked_up.receipt == receipt
    assert asyncio.run(authorized(resource, cmd)) == receipt
    with pytest.raises(ResourceOwnerConflict):
        asyncio.run(
            authorized(
                resource,
                cmd.model_copy(
                    update={"intent": cmd.intent.model_copy(update={"max_total_bytes": 2048})}
                ),
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(store.delete(artifact.id))
    asyncio.run(resource.release(receipt))
    asyncio.run(store.delete(artifact.id))
    assert isinstance(asyncio.run(resource.readback(cmd)), ExactUnavailable)


def test_released_acquisition_is_not_replayed_as_owned(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, cmd))
    asyncio.run(resource.release(receipt))
    with pytest.raises(ResourceOwnerUnavailable):
        asyncio.run(resource.acquire(cmd))
    asyncio.run(store.delete(artifact.id))


@pytest.mark.parametrize(
    "change",
    [
        lambda intent: intent.model_copy(update={"allowed_operations": ("read", "release")}),
        lambda intent: intent.model_copy(
            update={
                "selector": intent.selector.model_copy(
                    update={"resource": intent.selector.resource.model_copy(update={"revision": 2})}
                )
            }
        ),
        lambda intent: intent.model_copy(update={"max_materials": 2}),
        lambda intent: intent.model_copy(update={"max_manifest_bytes": 8192}),
        lambda intent: intent.model_copy(update={"max_total_bytes": 2048}),
        lambda intent: intent.model_copy(
            update={"policy": intent.policy.model_copy(update={"incarnation": "replacement"})}
        ),
    ],
)
def test_same_key_decision_field_changes_conflict_before_dispatch(tmp_path, change):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(authorized(resource, cmd))
    changed = cmd.model_copy(update={"intent": change(cmd.intent)})
    with pytest.raises((ResourceOwnerConflict, ResourceOwnerUnsupported)):

        async def run_changed():
            preparation = await resource.authorize(changed, permit=preparation_permit(changed))
            await resource.acquire(changed, preparation=preparation)

        asyncio.run(run_changed())
    asyncio.run(resource.release(asyncio.run(resource.readback(cmd)).receipt))
    asyncio.run(store.delete(artifact.id))


def test_unknown_selector_mode_fails_closed(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_owner = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_owner, artifact_store=store
    )
    cmd = command(resource_owner, artifact.id, store=store)
    bad = cmd.model_copy(
        update={
            "intent": cmd.intent.model_copy(
                update={"selector": cmd.intent.selector.model_copy(update={"mode": "subtree"})}
            )
        }
    )
    with pytest.raises(ResourceOwnerUnsupported):
        asyncio.run(resource.acquire(bad))


def test_raw_preparation_shaped_command_cannot_acquire(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    with pytest.raises(ResourceOwnerUnavailable):
        asyncio.run(resource.acquire(cmd))
    asyncio.run(store.delete(artifact.id))


def test_owner_without_registered_preparation_authority_fails_closed(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    with pytest.raises(ResourceOwnerUnsupported):
        _LocalArtifactResourceOwner(
            tmp_path / "owner", owner=owner("artifact-store"), artifact_store=store
        )
    asyncio.run(store.delete(artifact.id))


async def registered_resource(
    tmp_path,
    store,
    artifact,
    collaboration_store=None,
    *,
    required_settlement="exclusion",
    scope=None,
    folder_members=(),
    bounds=None,
):
    from tests.core.test_participant_identity import app, create, registration

    from cayu.artifacts.resources import MandateResourcePreparationReader
    from cayu.collaboration.memory import InMemoryCollaborationStore

    collaboration_store = collaboration_store or InMemoryCollaborationStore()
    application = app(collaboration_store, registration(scope=scope))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    participant = created.participants[0].reference
    base = LocalArtifactResourceOwner(
        tmp_path / "base", owner=initialized.owner, artifact_store=store
    )
    cmd = command(initialized.owner, artifact.id, store=store).model_copy(
        update={"operation": initialized.operation("acquisition")}
    )
    manifest = None
    if folder_members:
        manifest = FolderInputManifest(
            entries=tuple(
                FolderInputEntry(
                    path=f"input-{index}.txt",
                    git_mode="100644",
                    member=ArtifactInputMember(
                        resource=ObjectRef(
                            owner=initialized.owner,
                            kind="artifact",
                            object_id=metadata.id,
                            incarnation=hashlib.sha256(
                                canonical_durable_json_bytes(
                                    metadata.model_dump(mode="json"), "metadata"
                                )
                            ).hexdigest(),
                            revision=1,
                        ),
                        content_sha256=hashlib.sha256(content).hexdigest(),
                        metadata_sha256=hashlib.sha256(
                            canonical_durable_json_bytes(
                                metadata.model_dump(mode="json"), "metadata"
                            )
                        ).hexdigest(),
                        size_bytes=len(content),
                    ),
                )
                for index, (metadata, content) in enumerate(folder_members)
            )
        )
        selector = base.register_manifest(manifest)
        cmd = cmd.model_copy(
            update={
                "intent": cmd.intent.model_copy(
                    update={"selector": selector, "max_materials": len(folder_members)}
                )
            }
        )
    if bounds is not None:
        cmd = cmd.model_copy(update={"intent": cmd.intent.model_copy(update=bounds)})
    resolver, context = mandate_reader(base, cmd)
    cmd = cmd.model_copy(
        update={"initiator": cmd.initiator.model_copy(update={"mandate": context.mandate})}
    )
    raw = preparation_permit(cmd)
    permit = raw.model_copy(
        update={
            "intent": raw.intent.model_copy(
                update={
                    "limits": initialized.binding.limits,
                    "request": raw.intent.request.model_copy(
                        update={
                            "participant": participant,
                            "required_settlement": required_settlement,
                        }
                    ),
                }
            )
        }
    )
    # Release is separate current authority, not implied by prepare.
    resolver.resolution = resolver.resolution.model_copy(
        update={
            "principal": resolver.resolution.principal.model_copy(
                update={"actions": ("prepare", "release")}
            ),
            "chain": MandateChain(
                entries=(
                    resolver.resolution.chain.entries[0].model_copy(
                        update={"actions": ("prepare", "release")}
                    ),
                )
            ),
        }
    )
    reader = MandateResourcePreparationReader(
        owner=base,
        resolver=resolver,
        context=context,
        redactor=SecretRedactor(),
        registration=ObjectRef(
            owner=initialized.owner,
            kind="resource_receiver",
            object_id="receiver",
            incarnation="one",
            revision=1,
        ),
        policy=cmd.intent.policy,
        artifact_store=store,
        collaboration_store=collaboration_store,
        initialized=initialized,
        responsibilities=((cmd, permit),),
    )
    resource = _LocalArtifactResourceOwner(
        tmp_path / "mandated-owner",
        owner=initialized.owner,
        artifact_store=store,
        preparation_reader=reader,
    )
    if manifest is not None:
        assert resource.register_manifest(manifest) == cmd.intent.selector
    return resource, cmd, permit, resolver, collaboration_store, initialized, participant


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("kind", ["artifact", "folder"])
def test_public_acquisition_uses_registered_mandate_receiver(tmp_path, backend, request, kind):
    store, artifact = make_store(tmp_path)
    address = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        from cayu.artifacts.resources import MandateResourcePreparationReader
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        members = ((artifact, b"immutable"),)
        if kind == "folder":
            second = await store.put_bytes(
                b"second",
                artifact_id="art_" + "2" * 32,
                filename="second.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            members += ((second, b"second"),)
        collaboration = (
            SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
            if backend == "sqlite"
            else None
        )
        if backend == "postgres":
            from cayu.storage.collaboration_postgres import PostgresCollaborationStore
            from cayu.storage.migrations import SchemaMode

            collaboration = PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
        (
            resource,
            cmd,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(
            tmp_path,
            store,
            artifact,
            collaboration,
            folder_members=members if kind == "folder" else (),
        )
        try:
            preparation = await resource.authorize(cmd, permit=permit)
            receipt = await resource.acquire(cmd, preparation=preparation)
            assert receipt.material_count == len(members)
            assert receipt.total_bytes == sum(len(content) for _, content in members)
            assert set(receipt.material_ids) == {metadata.id for metadata, _ in members}
            assert (await resource.readback(cmd)).receipt == receipt
            for metadata, content in members:
                assert (await store.read_bytes(metadata.id)).content == content
                with pytest.raises(ValueError):
                    await store.delete(metadata.id)
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert len(obligations) == 1
            assert obligations[0].source_operation == cmd.operation
            # Reconstruct the resource receiver and both persistent journals,
            # retaining the exact host registration rather than a fake grant.
            old = resource._preparation_reader
            if backend == "sqlite":
                await collaboration.close()
                collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
            elif backend == "postgres":
                await collaboration.close()
                collaboration = PostgresCollaborationStore(address, schema_mode=SchemaMode.VALIDATE)
            reader = MandateResourcePreparationReader(
                owner=old._owner,
                resolver=resolver,
                context=old._context,
                redactor=SecretRedactor(),
                registration=old._registration,
                policy=old._policy,
                artifact_store=store,
                collaboration_store=collaboration,
                initialized=initialized,
                responsibilities=((cmd, permit),),
            )
            reopened = _LocalArtifactResourceOwner(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=reader,
            )
            lookup = await reopened.readback(cmd)
            assert isinstance(lookup, ExactMatch)
            assert lookup.receipt == receipt
            await reopened.release(lookup.receipt)
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
    "field",
    [
        "participant",
        "expected_lifecycle_revision",
        "admission_generation",
        "settlement_operation",
        "target_state",
        "source_operation",
    ],
)
def test_registered_receiver_rejects_caller_shaped_responsibility(tmp_path, field):
    store, artifact = make_store(tmp_path)

    async def run():
        (
            resource,
            cmd,
            permit,
            _resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(tmp_path, store, artifact)
        request = permit.intent.request
        values = {
            "participant": participant.model_copy(update={"incarnation": "other"}),
            "expected_lifecycle_revision": 2,
            "admission_generation": 2,
            "settlement_operation": initialized.operation("other-settlement"),
            "target_state": "existing",
            "source_operation": initialized.operation("unrelated"),
        }
        changed = permit.model_copy(
            update={
                "intent": permit.intent.model_copy(
                    update={"request": request.model_copy(update={field: values[field]})}
                )
            }
        )
        try:
            with pytest.raises(ResourceOwnerConflict):
                await resource.authorize(cmd, permit=changed)
            with resource._journal.locked() as journal:
                assert not journal["authorizations"]
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("replacement", [{"revision": 2}, {"object_id": "different-resolver"}])
def test_mandate_resolver_replacement_blocks_dispatch(tmp_path, replacement):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, cmd, permit, resolver, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            preparation = await resource.authorize(cmd, permit=permit)
            resolver.resolution = resolver.resolution.model_copy(
                update={
                    "principal": resolver.resolution.principal.model_copy(
                        update={"resolver": resolver.ref.model_copy(update=replacement)}
                    )
                }
            )
            with pytest.raises(ResourceOwnerUnavailable):
                await resource.acquire(cmd, preparation=preparation)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_registered_mandate_revocation_blocks_new_preparation(tmp_path):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, cmd, permit, resolver, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            resolver.revoked = True
            from cayu.collaboration.mandates import MandateDenied

            with pytest.raises(MandateDenied):
                await resource.authorize(cmd, permit=permit)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("denial", ["revoked", "other-principal"])
def test_public_release_denied_but_internal_cleanup_remains_available(tmp_path, denial):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, cmd, permit, resolver, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            receipt = await resource.acquire(
                cmd, preparation=await resource.authorize(cmd, permit=permit)
            )
            original_context = resource._preparation_reader._context
            if denial == "revoked":
                resolver.revoked = True
            else:
                resource._preparation_reader._context = original_context.model_copy(
                    update={"principal": "other"}
                )
            from cayu.collaboration.mandates import MandateDenied

            with pytest.raises((ResourceOwnerUnsupported, MandateDenied)):
                await resource.release(receipt)
            # Revocation denies new work; it does not authorize retiring live retention.
            assert await resource.reconcile() == ()
            with resource._journal.locked() as journal:
                assert journal["operations"][receipt.operation_digest]["stage"] == "owned"
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            resolver.revoked = False
            resource._preparation_reader._context = original_context
            await resource.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("ancestor_changed", [False, True])
def test_delegated_mandate_generations_are_bound_individually(tmp_path, ancestor_changed):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import MandateResourcePreparationReader

        resource, cmd, permit, resolver, collaboration, initialized, _ = await registered_resource(
            tmp_path, store, artifact
        )
        old = resource._preparation_reader
        root = resolver.resolution.chain.entries[0]
        child = root.model_copy(
            update={
                "reference": root.reference.model_copy(update={"object_id": "child"}),
                "parent": root.reference,
                "remaining_delegations": 0,
                "revocation_generation": 7,
            }
        )
        resolver.resolution = resolver.resolution.model_copy(
            update={"chain": MandateChain(entries=(root, child))}
        )
        context = old._context.model_copy(update={"mandate": child.reference})
        cmd = cmd.model_copy(
            update={"initiator": cmd.initiator.model_copy(update={"mandate": child.reference})}
        )
        permit = permit.model_copy(update={"initiator": cmd.initiator})
        resource._preparation_reader = MandateResourcePreparationReader(
            owner=old._owner,
            resolver=resolver,
            context=context,
            redactor=SecretRedactor(),
            registration=old._registration,
            policy=old._policy,
            artifact_store=store,
            collaboration_store=collaboration,
            initialized=initialized,
            responsibilities=((cmd, permit),),
        )
        try:
            prep = await resource.authorize(cmd, permit=permit)
            if ancestor_changed:
                resolver.resolution = resolver.resolution.model_copy(
                    update={
                        "chain": MandateChain(
                            entries=(root.model_copy(update={"revocation_generation": 2}), child)
                        )
                    }
                )
                with pytest.raises(ResourceOwnerUnavailable):
                    await resource.acquire(cmd, preparation=prep)
            else:
                receipt = await resource.acquire(cmd, preparation=prep)
                await resource.release(receipt)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["policy", "receiver", "context"])
def test_reopened_authority_rejects_same_revision_replacement(tmp_path, changed):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import MandateResourcePreparationReader

        resource, cmd, permit, resolver, collaboration, initialized, _ = await registered_resource(
            tmp_path, store, artifact
        )
        old = resource._preparation_reader
        try:
            prep = await resource.authorize(cmd, permit=permit)
            replacement = MandateResourcePreparationReader(
                owner=old._owner,
                resolver=resolver,
                context=old._context.model_copy(update={"principal": "other"})
                if changed == "context"
                else old._context,
                registration=old._registration.model_copy(update={"object_id": "replacement"})
                if changed == "receiver"
                else old._registration,
                policy=old._policy.model_copy(update={"object_id": "replacement"})
                if changed == "policy"
                else old._policy,
                redactor=SecretRedactor(),
                artifact_store=store,
                collaboration_store=collaboration,
                initialized=initialized,
                responsibilities=((cmd, permit),),
            )
            reopened = _LocalArtifactResourceOwner(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=replacement,
            )
            with pytest.raises((ResourceOwnerUnavailable, ResourceOwnerUnsupported)):
                await reopened.acquire(cmd, preparation=prep)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_expired_preparation_lease_cannot_dispatch(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)

    async def run():
        resource, cmd, permit, _resolver, collaboration, *_ = await registered_resource(
            tmp_path, store, artifact
        )
        try:
            prep = await resource.authorize(cmd, permit=permit)
            monkeypatch.setattr(
                "cayu.artifacts.resources.time.time", lambda: prep.lease.expires_at_ms / 1000 + 1
            )
            with pytest.raises(ResourceOwnerUnavailable):
                await resource.acquire(cmd, preparation=prep)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["before_registration", "registration", "pin", "settlement"])
@pytest.mark.parametrize("required_settlement", ["exclusion", "quiescence"])
@pytest.mark.parametrize("kind", ["artifact", "folder"])
def test_collaboration_lost_ack_reconciles_without_new_responsibility(
    tmp_path, monkeypatch, phase, required_settlement, kind
):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        members = ((artifact, b"immutable"),)
        if kind == "folder":
            second = await store.put_bytes(
                b"second",
                artifact_id="art_" + "2" * 32,
                filename="second.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="resource",
            )
            members += ((second, b"second"),)
        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        (
            resource,
            cmd,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_resource(
            tmp_path,
            store,
            artifact,
            collaboration,
            required_settlement=required_settlement,
            folder_members=members if kind == "folder" else (),
        )
        method = {
            "before_registration": "_register_permit",
            "registration": "_register_permit",
            "pin": "_pin_resource",
            "settlement": "_exclude_permit",
        }[phase]
        target = store if phase == "pin" else collaboration
        original = getattr(target, method)
        failed = False

        async def lost_ack(*args, **kwargs):
            nonlocal failed
            if phase == "before_registration":
                raise OSError("lost acknowledgement")
            result = await original(*args, **kwargs)
            if not failed:
                failed = True
                raise OSError("lost acknowledgement")
            return result

        monkeypatch.setattr(target, method, lost_ack)
        try:
            prep = await resource.authorize(cmd, permit=permit)
            if phase in {"before_registration", "registration", "pin"}:
                with pytest.raises(OSError, match="lost acknowledgement"):
                    await resource.acquire(cmd, preparation=prep)
                if phase == "pin":
                    with pytest.raises(ValueError):
                        await store.delete(artifact.id)
            else:
                receipt = await resource.acquire(cmd, preparation=prep)
                with pytest.raises(OSError, match="lost acknowledgement"):
                    await resource.release(receipt)
            resolver.revoked = True
            # Reopen both durable owners; no in-process collaboration state survives.
            from cayu.artifacts.resources import MandateResourcePreparationReader

            old = resource._preparation_reader
            await collaboration.close()
            collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
            reader = MandateResourcePreparationReader(
                owner=old._owner,
                resolver=resolver,
                context=old._context,
                registration=old._registration,
                policy=old._policy,
                artifact_store=store,
                collaboration_store=collaboration,
                initialized=initialized,
                responsibilities=((cmd, permit),),
                redactor=SecretRedactor(),
            )
            reopened = _LocalArtifactResourceOwner(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=reader,
            )
            await reopened.reconcile()
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=False,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert len(obligations) == (0 if phase == "before_registration" else 1)
            if obligations:
                assert obligations[0].state == "settled"
                assert obligations[0].outcome == "quiescent"
            from cayu.artifacts.resources import _reserved_bytes

            with reopened._journal.locked() as journal:
                assert _reserved_bytes(journal) == 0
            if phase == "before_registration":
                from cayu.collaboration._contracts import CollaborationConflict

                with pytest.raises(CollaborationConflict):
                    await collaboration._register_permit(
                        initialized, permit, redactor=SecretRedactor()
                    )
            await reopened.reconcile()
            for metadata, _ in members:
                await store.delete(metadata.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_preparation_receipt_is_bound_to_runtime_lease(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    preparation = asyncio.run(resource.authorize(cmd, permit=preparation_permit(cmd)))
    forged = preparation.model_copy(
        update={"lease": preparation.lease.model_copy(update={"receiver": owner("other")})}
    )
    with pytest.raises(ResourceOwnerConflict):
        asyncio.run(resource.acquire(cmd, preparation=forged))
    asyncio.run(store.delete(artifact.id))


def test_rejected_preparation_is_diagnostic_silent(tmp_path, caplog, capsys):
    import warnings

    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ResourceOwnerUnavailable):
            asyncio.run(resource.acquire(command(resource_ref, artifact.id, store=store)))
    assert not emitted
    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_reopen_reads_durable_owner_state(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_owner = owner("artifact-store")
    cmd = command(resource_owner, artifact.id, store=store)
    first = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_owner, artifact_store=store
    )
    receipt = asyncio.run(authorized(first, cmd))
    second = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_owner, artifact_store=store
    )
    looked_up = asyncio.run(second.readback(cmd))
    assert isinstance(looked_up, ExactMatch)
    assert looked_up.receipt == receipt


def test_reopen_with_different_owner_or_store_fails_closed(tmp_path: Path):
    store, _artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    LocalArtifactResourceOwner(tmp_path / "owner", owner=resource_ref, artifact_store=store)
    with pytest.raises(ResourceOwnerUnavailable):
        LocalArtifactResourceOwner(
            tmp_path / "owner", owner=owner("replacement"), artifact_store=store
        )
    other_store = LocalArtifactStore(tmp_path / "other-artifacts")
    with pytest.raises(ResourceOwnerUnavailable):
        LocalArtifactResourceOwner(
            tmp_path / "owner", owner=resource_ref, artifact_store=other_store
        )


def test_initialized_journal_without_identity_fails_closed(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(authorized(resource, cmd))
    journal_path = tmp_path / "owner" / "resource-owner.json"
    journal = json.loads(journal_path.read_text())
    journal.pop("owner")
    journal.pop("artifact_store")
    journal_path.write_text(json.dumps(journal))
    with pytest.raises(ResourceOwnerUnavailable):
        LocalArtifactResourceOwner(tmp_path / "owner", owner=resource_ref, artifact_store=store)


def test_transfer_pins_destination_before_source_release(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    source_ref = owner("source")
    destination_ref = owner("destination")
    source = LocalArtifactResourceOwner(
        tmp_path / "source-owner", owner=source_ref, artifact_store=store
    )
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination-owner", owner=destination_ref, artifact_store=store
    )
    source_cmd = command(source_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(source, source_cmd))
    transfer_intent = ResourceTransferIntent(
        receipt=receipt,
        destination=destination_ref,
        cleanup_owner=source_ref,
        acceptance_generation=1,
        expected_material_commitment=receipt.content_commitment,
    )
    transfer = ResourceTransferCommand(
        operation=OperationRef(
            application_scope="app",
            namespace_incarnation="namespace",
            generation=1,
            caller_key="transfer",
        ),
        source=source_ref,
        destination=destination_ref,
        initiator=receipt.command.initiator,
        intent=transfer_intent,
    )
    accepted = asyncio.run(destination.accept_transfer(transfer, source_owner=source))
    assert accepted.stage == "accepted"
    with pytest.raises(ValueError):
        asyncio.run(store.delete(artifact.id))
    asyncio.run(source.release_transferred_source(accepted, destination_owner=destination))
    # The destination pin still protects the material after source release.
    with pytest.raises(ValueError):
        asyncio.run(store.delete(artifact.id))
    assert asyncio.run(destination.accept_transfer(transfer, source_owner=source)) == accepted


def _transfer_command(receipt, source_ref, destination_ref):
    return ResourceTransferCommand(
        operation=OperationRef(
            application_scope=source_ref.application_scope,
            namespace_incarnation="namespace",
            generation=1,
            caller_key="transfer",
        ),
        source=source_ref,
        destination=destination_ref,
        initiator=receipt.command.initiator,
        intent=ResourceTransferIntent(
            receipt=receipt,
            destination=destination_ref,
            cleanup_owner=source_ref,
            acceptance_generation=1,
            expected_material_commitment=receipt.content_commitment,
        ),
    )


async def registered_transfer(tmp_path, store, artifact, source, receipt, *, include=True):
    from cayu.artifacts.resources import MandateResourcePreparationReader
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

    collaboration = SQLiteCollaborationStore(tmp_path / "destination.sqlite")
    destination, _, _, resolver, _, initialized, participant = await registered_resource(
        tmp_path, store, artifact, collaboration, scope=source.owner.application_scope
    )
    transfer = _transfer_command(receipt, source.owner, destination.owner).model_copy(
        update={"operation": initialized.operation("transfer")}
    )
    raw = preparation_permit(transfer)
    permit = raw.model_copy(
        update={
            "intent": raw.intent.model_copy(
                update={
                    "limits": initialized.binding.limits,
                    "request": raw.intent.request.model_copy(
                        update={"participant": participant, "required_settlement": "quiescence"}
                    ),
                }
            )
        }
    )
    resolver.resolution = resolver.resolution.model_copy(
        update={
            "chain": MandateChain(
                entries=(
                    resolver.resolution.chain.entries[0].model_copy(
                        update={"resources": (receipt.command.intent.selector,)}
                    ),
                )
            )
        }
    )
    old = destination._preparation_reader
    destination._preparation_reader = MandateResourcePreparationReader(
        owner=old._owner,
        resolver=resolver,
        context=old._context,
        redactor=SecretRedactor(),
        registration=old._registration,
        policy=receipt.command.intent.policy,
        artifact_store=store,
        collaboration_store=collaboration,
        initialized=initialized,
        responsibilities=(),
        transfers=((transfer, permit),) if include else (),
    )
    return destination, transfer, permit, resolver, collaboration, initialized, participant


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "readback_revoked",
        "missing",
        "revoked",
        "changed_initiator",
        "changed_generation",
    ],
)
def test_transfer_requires_exact_registered_destination_authority(tmp_path, monkeypatch, case):
    store, artifact = make_store(tmp_path)

    async def run():
        source, cmd, source_permit, _, source_store, *_ = await registered_resource(
            tmp_path / "source", store, artifact
        )
        receipt = await source.acquire(
            cmd, preparation=await source.authorize(cmd, permit=source_permit)
        )
        (
            destination,
            transfer,
            _,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_transfer(
            tmp_path / "destination", store, artifact, source, receipt, include=case != "missing"
        )
        pins = []
        original_pin = store._pin_resource

        async def _pin_resource(artifact_id, *, owner, dispatch):
            pins.append(owner)
            return await original_pin(artifact_id, owner=owner, dispatch=dispatch)

        monkeypatch.setattr(store, "_pin_resource", _pin_resource)
        from cayu.collaboration.mandates import MandateDenied

        try:
            if case == "revoked":
                resolver.revoked = True
            elif case == "changed_generation":
                transfer = transfer.model_copy(
                    update={
                        "intent": transfer.intent.model_copy(update={"acceptance_generation": 2})
                    }
                )
            elif case == "changed_initiator":
                transfer = transfer.model_copy(
                    update={
                        "initiator": transfer.initiator.model_copy(
                            update={"principal": "different"}
                        )
                    }
                )
            if case in {"success", "readback_revoked"}:
                accepted = await destination.accept_transfer(transfer, source_owner=source)
                assert accepted.stage == "accepted"
                assert isinstance(await destination.read_transfer(transfer), ExactMatch)
                if case == "readback_revoked":
                    resolver.revoked = True
                    from cayu.collaboration._contracts import ExactUnavailable

                    assert isinstance(await destination.read_transfer(transfer), ExactUnavailable)
                assert len(pins) == 1
                _, obligations = await collaboration.scan_obligations(
                    initialized,
                    participant,
                    after=0,
                    limit=64,
                    pending_only=True,
                    retention_revision=None,
                    redactor=SecretRedactor(),
                )
                assert len(obligations) == 1
                assert obligations[0].source_operation == transfer.operation
                await source.release_transferred_source(accepted, destination_owner=destination)
                with pytest.raises(ValueError):
                    await store.delete(artifact.id)
                resolver.revoked = False
                await destination.release_transfer(accepted)
                _, obligations = await collaboration.scan_obligations(
                    initialized,
                    participant,
                    after=0,
                    limit=64,
                    pending_only=True,
                    retention_revision=None,
                    redactor=SecretRedactor(),
                )
                assert obligations == ()
                await store.delete(artifact.id)
            else:
                with pytest.raises(
                    (ResourceOwnerUnsupported, ResourceOwnerConflict, MandateDenied)
                ):
                    await destination.accept_transfer(transfer, source_owner=source)
                assert pins == []
                with destination._journal.locked() as journal:
                    assert journal["transfers"] == {}
                await source.release(receipt)
                await store.delete(artifact.id)
        finally:
            await collaboration.close()
            await source_store.close()

    asyncio.run(run())


@pytest.mark.parametrize("revoke_after", [1, 2])
def test_transfer_revocation_retains_partial_pins_until_reopened_cleanup(
    tmp_path, monkeypatch, revoke_after
):
    store, first = make_store(tmp_path)

    async def run():
        from cayu.artifacts.resources import MandateResourcePreparationReader, _reserved_bytes
        from cayu.collaboration.mandates import MandateDenied
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        second = await store.put_bytes(
            b"second",
            artifact_id="art_" + "2" * 32,
            filename="second.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
        source = LocalArtifactResourceOwner(
            tmp_path / "source", owner=owner("source"), artifact_store=store
        )

        def member(artifact, content):
            return ArtifactInputMember(
                resource=ObjectRef(
                    owner=source.owner,
                    kind="artifact",
                    object_id=artifact.id,
                    incarnation=hashlib.sha256(
                        canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
                    ).hexdigest(),
                    revision=1,
                ),
                content_sha256=hashlib.sha256(content).hexdigest(),
                metadata_sha256=hashlib.sha256(
                    canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
                ).hexdigest(),
                size_bytes=len(content),
            )

        selector = source.register_manifest(
            FolderInputManifest(
                entries=(
                    FolderInputEntry(
                        path="first", git_mode="100644", member=member(first, b"immutable")
                    ),
                    FolderInputEntry(
                        path="second", git_mode="100644", member=member(second, b"second")
                    ),
                )
            )
        )
        base = command(source.owner, first.id, store=store)
        cmd = base.model_copy(
            update={
                "intent": base.intent.model_copy(update={"selector": selector, "max_materials": 2})
            }
        )
        receipt = await authorized(source, cmd)
        (
            destination,
            transfer,
            permit,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_transfer(tmp_path / "destination", store, first, source, receipt)
        original_pin = store._pin_resource
        dispatched = []

        async def _pin_resource(artifact_id, *, owner, dispatch):
            await original_pin(artifact_id, owner=owner, dispatch=dispatch)
            dispatched.append(artifact_id)
            if len(dispatched) == revoke_after:
                resolver.revoked = True

        monkeypatch.setattr(store, "_pin_resource", _pin_resource)
        try:
            with pytest.raises(MandateDenied):
                await destination.accept_transfer(transfer, source_owner=source)
            assert len(dispatched) == revoke_after
            digest = resource_operation_digest(transfer)
            with destination._journal.locked() as journal:
                assert journal["transfers"][digest]["stage"] == "uncertain"
                assert _reserved_bytes(journal) == receipt.total_bytes
                assert journal["authorizations"][digest]["receipt"]["lease"]["expires_at_ms"] > int(
                    time.time() * 1000
                )
            await source.release(receipt)
            with pytest.raises(ValueError):
                await store.delete(first.id)
            old = destination._preparation_reader
            await collaboration.close()
            collaboration = SQLiteCollaborationStore(
                tmp_path / "destination" / "destination.sqlite"
            )
            reader = MandateResourcePreparationReader(
                owner=old._owner,
                resolver=resolver,
                context=old._context,
                redactor=SecretRedactor(),
                registration=old._registration,
                policy=old._policy,
                artifact_store=store,
                collaboration_store=collaboration,
                initialized=initialized,
                responsibilities=(),
                transfers=((transfer, permit),),
            )
            reopened = _LocalArtifactResourceOwner(
                tmp_path / "destination" / "mandated-owner",
                owner=destination.owner,
                artifact_store=store,
                preparation_reader=reader,
            )
            assert await reopened.reconcile(source_owners=(source,)) == (digest,)
            assert len(dispatched) == revoke_after
            with reopened._journal.locked() as journal:
                assert _reserved_bytes(journal) == 0
                assert journal["transfers"][digest]["stage"] == "released"
            _, obligations = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=False,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert len(obligations) == 1
            assert obligations[0].state == "settled"
            assert obligations[0].outcome == "quiescent"
            assert await reopened.reconcile(source_owners=(source,)) == ()
            await store.delete(first.id)
            await store.delete(second.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_cancelled_transfer_keeps_destination_responsibility_fenced(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)

    async def run():
        source = LocalArtifactResourceOwner(
            tmp_path / "source", owner=owner("source"), artifact_store=store
        )
        receipt = await authorized(source, command(source.owner, artifact.id, store=store))
        (
            destination,
            transfer,
            _,
            resolver,
            collaboration,
            initialized,
            participant,
        ) = await registered_transfer(tmp_path / "destination", store, artifact, source, receipt)
        entered, finish = asyncio.Event(), asyncio.Event()
        original_pin = store._pin_resource
        calls = []

        async def _pin_resource(artifact_id, *, owner, dispatch):
            await original_pin(artifact_id, owner=owner, dispatch=dispatch)
            calls.append(owner)
            entered.set()
            await finish.wait()

        monkeypatch.setattr(store, "_pin_resource", _pin_resource)
        task = asyncio.create_task(destination.accept_transfer(transfer, source_owner=source))
        try:
            async with asyncio.timeout(10):
                await entered.wait()
            task.cancel()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("Transfer cancellation was lost")
            assert task.cancelled()
            assert task.cancelling() == 2
            resolver.revoked = True
            with pytest.raises(ResourceOwnerUnavailable):
                await destination.reconcile(source_owners=(source,))
            with pytest.raises(ResourceOwnerUnavailable):
                await source.release(receipt)
            _, pending = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert len(pending) == 1
            finish.set()
            with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                await destination.drain()
            await destination.reconcile(source_owners=(source,))
            await source.release(receipt)
            await store.delete(artifact.id)
            assert len(calls) == 1
            _, pending = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=True,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            assert pending == ()
        finally:
            finish.set()
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("capacity", [13, 14, 15])
def test_event_reservations_aggregate_acquisition_and_transfer(tmp_path, monkeypatch, capacity):
    import cayu.artifacts.resources as resources

    monkeypatch.setattr(resources, "RESOURCE_MAX_EVENTS", capacity)
    store, artifact = make_store(tmp_path)
    source = LocalArtifactResourceOwner(
        tmp_path / "source", owner=owner("source"), artifact_store=store
    )
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination", owner=owner("destination"), artifact_store=store
    )

    async def run():
        source_receipt = await authorized(source, command(source.owner, artifact.id, store=store))
        local_receipt = await authorized(
            destination, command(destination.owner, artifact.id, store=store)
        )
        transfer = _transfer_command(source_receipt, source.owner, destination.owner)
        with destination._journal.locked() as journal:
            assert resources._event_usage(journal) == 7
        if capacity < 14:
            with pytest.raises(resources.ResourceOwnerError, match="event reservation"):
                await destination.accept_transfer(transfer, source_owner=source)
            with destination._journal.locked() as journal:
                assert journal["transfers"] == {}
            await destination.release(local_receipt)
        else:
            accepted = await destination.accept_transfer(transfer, source_owner=source)
            with destination._journal.locked() as journal:
                assert resources._event_usage(journal) == 14
            extra = command(destination.owner, artifact.id, store=store, key="extra")
            await destination.authorize(extra, permit=preparation_permit(extra))
            with pytest.raises(resources.ResourceOwnerError, match="event reservation"):
                await destination.acquire(extra)
            original_release = store._release_resource_pin

            async def fail_release(*args, **kwargs):
                raise OSError("cleanup unavailable")

            monkeypatch.setattr(store, "_release_resource_pin", fail_release)
            for _ in range(20):
                with pytest.raises(OSError):
                    await destination.release(local_receipt)
                with pytest.raises(OSError):
                    await destination.release_transfer(accepted)
            with destination._journal.locked() as journal:
                assert resources._event_usage(journal) == 14
                assert len(journal["events"]) == 6
            monkeypatch.setattr(store, "_release_resource_pin", original_release)
            reopened = LocalArtifactResourceOwner(
                tmp_path / "destination", owner=destination.owner, artifact_store=store
            )
            assert len(await reopened.reconcile()) == 2
            with reopened._journal.locked() as journal:
                assert len(journal["events"]) == 10
                assert resources._event_usage(journal) == 10
                assert resources._reserved_bytes(journal) == 0
        await source.release(source_receipt)
        await store.delete(artifact.id)

    asyncio.run(run())


def test_retry_diagnostics_cannot_spend_terminal_event_reservations(tmp_path, monkeypatch):
    import cayu.artifacts.resources as resources

    monkeypatch.setattr(resources, "RESOURCE_MAX_EVENTS", 7)
    store, artifact = make_store(tmp_path)
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=owner("resource"), artifact_store=store
    )
    cmd = command(resource.owner, artifact.id, store=store)
    original_read = store.read_bytes

    async def fail_read(*args, **kwargs):
        raise OSError("read unavailable")

    monkeypatch.setattr(store, "read_bytes", fail_read)

    async def run():
        await resource.authorize(cmd, permit=preparation_permit(cmd))
        with pytest.raises(OSError):
            await resource.acquire(cmd)
        for _ in range(20):
            assert await resource.reconcile() == ()
        with resource._journal.locked() as journal:
            assert len(journal["events"]) == 2
            assert resources._event_usage(journal) == 7
        monkeypatch.setattr(store, "read_bytes", original_read)
        await resource.reconcile()
        lookup = await resource.readback(cmd)
        assert isinstance(lookup, ExactMatch)
        await resource.release(lookup.receipt)
        with resource._journal.locked() as journal:
            assert len(journal["events"]) == 6
            assert resources._event_usage(journal) == 6
        await store.delete(artifact.id)

    asyncio.run(run())


def test_transfer_process_loss_reconciles_destination_pin(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    source_ref, destination_ref = owner("source"), owner("destination")
    source = LocalArtifactResourceOwner(tmp_path / "source", owner=source_ref, artifact_store=store)
    source_cmd = command(source_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(source, source_cmd))
    transfer = _transfer_command(receipt, source_ref, destination_ref)
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_crash_transfer,
        args=(
            str(tmp_path / "source"),
            str(tmp_path / "destination"),
            str(tmp_path / "artifacts"),
            source_ref,
            destination_ref,
            transfer,
        ),
    )
    child.start()
    child.join(20)
    assert child.exitcode == 0
    destination = LocalArtifactResourceOwner(
        tmp_path / "destination", owner=destination_ref, artifact_store=store
    )
    assert asyncio.run(destination.reconcile(source_owners=(source,))) == (
        resource_operation_digest(transfer),
    )
    accepted = asyncio.run(destination.read_transfer(transfer))
    assert isinstance(accepted, ExactMatch)
    asyncio.run(source.release_transferred_source(accepted.receipt, destination_owner=destination))
    asyncio.run(destination.release_transfer(accepted.receipt))
    asyncio.run(store.delete(artifact.id))


@pytest.mark.parametrize("delete_before_recovery", [False, True])
def test_release_process_loss_reconciles_source_pin(tmp_path: Path, delete_before_recovery):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    acquire_cmd = command(resource_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, acquire_cmd))
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_crash_release,
        args=(str(tmp_path / "owner"), str(tmp_path / "artifacts"), resource_ref, receipt),
    )
    child.start()
    child.join(20)
    assert child.exitcode == 0
    if delete_before_recovery:
        asyncio.run(store.delete(artifact.id))
    reopened = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    assert asyncio.run(reopened.reconcile()) == (receipt.operation_digest,)
    asyncio.run(store.delete(artifact.id))


def test_folder_manifest_pins_all_members_and_reopens(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    owner_ref = owner("artifact-store")
    first = asyncio.run(
        store.put_bytes(
            b"one",
            artifact_id="art_" + "1" * 32,
            filename="one",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    second = asyncio.run(
        store.put_bytes(
            b"two",
            artifact_id="art_" + "2" * 32,
            filename="two",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )

    def member(artifact, data):
        return ArtifactInputMember(
            resource=ObjectRef(
                owner=owner_ref,
                kind="artifact",
                object_id=artifact.id,
                incarnation=hashlib.sha256(
                    canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
                ).hexdigest(),
                revision=1,
            ),
            content_sha256=hashlib.sha256(data).hexdigest(),
            metadata_sha256=hashlib.sha256(
                canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
            ).hexdigest(),
            size_bytes=len(data),
        )

    manifest = FolderInputManifest(
        entries=(
            FolderInputEntry(path="one.txt", git_mode="100644", member=member(first, b"one")),
            FolderInputEntry(path="two.txt", git_mode="100644", member=member(second, b"two")),
        )
    )
    owner_instance = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=owner_ref, artifact_store=store
    )
    selector = owner_instance.register_manifest(manifest)
    cmd = command(owner_ref, selector.resource.object_id).model_copy(
        update={
            "intent": command(owner_ref, selector.resource.object_id).intent.model_copy(
                update={"selector": selector, "max_materials": 2}
            )
        }
    )
    receipt = asyncio.run(authorized(owner_instance, cmd))
    assert receipt.material_count == 2
    reopened = LocalArtifactResourceOwner(tmp_path / "owner", owner=owner_ref, artifact_store=store)
    lookup = asyncio.run(reopened.readback(cmd))
    assert isinstance(lookup, ExactMatch)
    asyncio.run(reopened.release(lookup.receipt))
    asyncio.run(store.delete(first.id))
    asyncio.run(store.delete(second.id))


def test_real_task_cancellation_retains_pending_acquisition(tmp_path: Path):
    store = BlockingReadStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"immutable",
            artifact_id="art_" + "1" * 32,
            filename="input.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(resource.authorize(cmd, permit=preparation_permit(cmd)))

    async def run():
        task = asyncio.create_task(resource.acquire(cmd))
        await store.started.wait()
        task.cancel("operator cancellation")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        store.release_read.set()
        with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
            await resource.drain()
        assert isinstance(await resource.readback(cmd), ExactUnavailable)
        with resource._journal.locked() as journal:
            assert journal["operations"][resource_operation_digest(cmd)]["stage"] == "released"

    asyncio.run(run())
    asyncio.run(store.delete(artifact.id))


def test_partial_folder_failure_retains_members_until_reconcile(tmp_path: Path):
    store = FailingReadStore(tmp_path / "artifacts")
    owner_ref = owner("artifact-store")
    first = asyncio.run(
        store.put_bytes(
            b"one",
            artifact_id="art_" + "1" * 32,
            filename="one",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    second = asyncio.run(
        store.put_bytes(
            b"two",
            artifact_id="art_" + "2" * 32,
            filename="two",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )

    def member(artifact, data):
        return ArtifactInputMember(
            resource=ObjectRef(
                owner=owner_ref,
                kind="artifact",
                object_id=artifact.id,
                incarnation=hashlib.sha256(
                    canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
                ).hexdigest(),
                revision=1,
            ),
            content_sha256=hashlib.sha256(data).hexdigest(),
            metadata_sha256=hashlib.sha256(
                canonical_durable_json_bytes(artifact.model_dump(mode="json"), "metadata")
            ).hexdigest(),
            size_bytes=len(data),
        )

    resource = LocalArtifactResourceOwner(tmp_path / "owner", owner=owner_ref, artifact_store=store)
    selector = resource.register_manifest(
        FolderInputManifest(
            entries=(
                FolderInputEntry(path="one", git_mode="100644", member=member(first, b"one")),
                FolderInputEntry(path="two", git_mode="100644", member=member(second, b"two")),
            )
        )
    )
    cmd = command(owner_ref, first.id, store=store).model_copy(
        update={
            "intent": command(owner_ref, first.id, store=store).intent.model_copy(
                update={"selector": selector, "max_materials": 2}
            )
        }
    )
    store.fail_reads = True
    asyncio.run(resource.authorize(cmd, permit=preparation_permit(cmd)))
    with pytest.raises(OSError):
        asyncio.run(resource.acquire(cmd))
    with pytest.raises(ValueError):
        asyncio.run(store.delete(first.id))
    with pytest.raises(ValueError):
        asyncio.run(store.delete(second.id))
    store.fail_reads = False
    assert asyncio.run(resource.reconcile()) == (resource_operation_digest(cmd),)
    lookup = asyncio.run(resource.readback(cmd))
    assert isinstance(lookup, ExactMatch)
    asyncio.run(resource.release(lookup.receipt))
    asyncio.run(store.delete(first.id))
    asyncio.run(store.delete(second.id))


def test_pending_acquisition_survives_process_loss_and_reconcile(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"crash-safe",
            artifact_id="art_" + "3" * 32,
            filename="input.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    owner_ref = owner("artifact-store")
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_crash_after_pending,
        args=(str(tmp_path / "owner"), str(tmp_path / "artifacts"), owner_ref, artifact.id),
    )
    child.start()
    child.join(20)
    assert child.exitcode == 0
    resource = LocalArtifactResourceOwner(tmp_path / "owner", owner=owner_ref, artifact_store=store)
    cmd = command(owner_ref, artifact.id, store=store)
    with pytest.raises(ResourceOwnerUnavailable):
        asyncio.run(resource.acquire(cmd))
    assert asyncio.run(resource.reconcile()) == (resource_operation_digest(cmd),)
    receipt = asyncio.run(resource.readback(cmd)).receipt
    asyncio.run(resource.release(receipt))


def test_process_loss_cleanup_does_not_require_current_authority(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    owner_ref = owner("artifact-store")
    context = multiprocessing.get_context("fork")
    child = context.Process(
        target=_crash_after_pending,
        args=(str(tmp_path / "owner"), str(tmp_path / "artifacts"), owner_ref, artifact.id),
    )
    child.start()
    child.join(20)
    assert child.exitcode == 0
    unavailable = _UnavailablePreparationReader(owner_ref)
    resource = _LocalArtifactResourceOwner(
        tmp_path / "owner",
        owner=owner_ref,
        artifact_store=store,
        preparation_reader=unavailable,
    )
    assert asyncio.run(resource.reconcile()) == (
        resource_operation_digest(command(owner_ref, artifact.id, store=store)),
    )
    asyncio.run(store.delete(artifact.id))


def test_unqualified_store_is_rejected_before_selection(tmp_path: Path):
    with pytest.raises(ResourceOwnerUnsupported):
        LocalArtifactResourceOwner(
            tmp_path / "owner",
            owner=owner("s3"),
            artifact_store=S3ArtifactStore("not-used"),
        )


def test_reserved_bytes_capacity_is_bounded_and_released(tmp_path, monkeypatch):
    import cayu.artifacts.resources as resources

    store, first = make_store(tmp_path)
    second = asyncio.run(
        store.put_bytes(
            b"x",
            artifact_id="art_" + "6" * 32,
            filename="second.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    monkeypatch.setattr(resources, "RESOURCE_MAX_RESERVED_BYTES", 10)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    first_cmd = command(resource_ref, first.id, store=store).model_copy(
        update={
            "intent": command(resource_ref, first.id, store=store).intent.model_copy(
                update={"max_total_bytes": 10}
            )
        }
    )
    second_cmd = command(resource_ref, second.id, store=store, key="second").model_copy(
        update={
            "intent": command(resource_ref, second.id, store=store, key="second").intent.model_copy(
                update={"max_total_bytes": 5}
            )
        }
    )
    asyncio.run(authorized(resource, first_cmd))
    with pytest.raises(ResourceOwnerError):
        asyncio.run(authorized(resource, second_cmd))
    first_receipt = asyncio.run(resource.readback(first_cmd)).receipt
    asyncio.run(resource.release(first_receipt))
    asyncio.run(authorized(resource, second_cmd))
    second_receipt = asyncio.run(resource.readback(second_cmd)).receipt
    asyncio.run(resource.release(second_receipt))
    asyncio.run(store.delete(first.id))
    asyncio.run(store.delete(second.id))


def test_pinned_delete_is_fenced_across_processes(tmp_path):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, cmd))
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=_delete_from_process,
        args=(str(tmp_path / "artifacts"), artifact.id, queue),
    )
    process.start()
    process.join(20)
    assert process.exitcode == 0
    assert queue.get(timeout=2) == "ValueError"
    asyncio.run(resource.release(receipt))
    asyncio.run(store.delete(artifact.id))


def test_release_interruption_retains_releasing_state_for_retry(tmp_path: Path):
    store = FailReleaseStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"release-safe",
            artifact_id="art_" + "4" * 32,
            filename="input.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    acquire_cmd = command(resource_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, acquire_cmd))
    with pytest.raises(OSError):
        asyncio.run(resource.release(receipt))
    assert isinstance(asyncio.run(resource.readback(receipt.command)), ExactUnavailable)
    asyncio.run(resource.release(receipt))
    asyncio.run(store.delete(artifact.id))


def test_release_rejects_serialized_receipt_material_substitution(tmp_path: Path):
    store, artifact = make_store(tmp_path)
    other = asyncio.run(
        store.put_bytes(
            b"other",
            artifact_id="art_" + "5" * 32,
            filename="other.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="resource",
        )
    )
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    acquire_cmd = command(resource_ref, artifact.id, store=store)
    receipt = asyncio.run(authorized(resource, acquire_cmd))
    forged = receipt.model_copy(update={"material_ids": (other.id,)})
    with pytest.raises(ResourceOwnerConflict):
        asyncio.run(resource.release(forged))
    asyncio.run(resource.release(receipt))
    asyncio.run(store.delete(artifact.id))
    asyncio.run(store.delete(other.id))


@pytest.mark.parametrize("state", ["owned", "releasing", "released"])
def test_release_replays_require_original_receipt_after_reopening(tmp_path, monkeypatch, state):
    store, artifact = make_store(tmp_path)

    async def run():
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        resource, cmd, permit, _, _, initialized, participant = await registered_resource(
            tmp_path, store, artifact, collaboration
        )
        original_release = store._release_resource_pin

        async def fail_release(*args, **kwargs):
            raise OSError("blocked release")

        try:
            receipt = await resource.acquire(
                cmd, preparation=await resource.authorize(cmd, permit=permit)
            )
            if state == "releasing":
                monkeypatch.setattr(store, "_release_resource_pin", fail_release)
                with pytest.raises(OSError):
                    await resource.release(receipt)
                monkeypatch.setattr(store, "_release_resource_pin", original_release)
            elif state == "released":
                await resource.release(receipt)
            reopened = _LocalArtifactResourceOwner(
                tmp_path / "mandated-owner",
                owner=resource.owner,
                artifact_store=store,
                preparation_reader=resource._preparation_reader,
            )
            calls = []

            async def tracked_release(*args, **kwargs):
                calls.append((args, kwargs))
                return await original_release(*args, **kwargs)

            monkeypatch.setattr(store, "_release_resource_pin", tracked_release)
            snapshot = reopened._journal.path.read_bytes()
            before = await collaboration.scan_obligations(
                initialized,
                participant,
                after=0,
                limit=64,
                pending_only=False,
                retention_revision=None,
                redactor=SecretRedactor(),
            )
            # All substitutions remain structurally valid; exact owner receipt
            # comparison, not model-shape validation, must reject them.
            changes = (
                {"receipt_id": "different"},
                {"material_ids": ("art_" + "9" * 32,)},
                {"content_commitment": "sha256:" + "9" * 64},
                {"manifest_commitment": "9" * 64},
                {"total_bytes": receipt.total_bytes + 1},
                {"stage": "releasing"},
                {"stage": "released"},
            )
            for update in changes:
                raw = receipt.model_dump(mode="json")
                raw.update(update)
                conflicting = ResourceAcquisitionReceipt.model_validate(raw)
                with pytest.raises(ResourceOwnerConflict):
                    await reopened.release(conflicting)
                assert calls == []
                assert reopened._journal.path.read_bytes() == snapshot
                assert (
                    await collaboration.scan_obligations(
                        initialized,
                        participant,
                        after=0,
                        limit=64,
                        pending_only=False,
                        retention_revision=None,
                        redactor=SecretRedactor(),
                    )
                    == before
                )
            if state != "released":
                with pytest.raises(ValueError):
                    await store.delete(artifact.id)
            exact = ResourceAcquisitionReceipt.model_validate_json(receipt.model_dump_json())
            await reopened.release(exact)
            count = len(calls)
            await reopened.release(exact)
            assert len(calls) == count == (0 if state == "released" else 1)
            await store.delete(artifact.id)
        finally:
            await collaboration.close()

    asyncio.run(run())


def test_commit_then_raise_replays_durable_receipt_without_reacquire(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(resource.authorize(cmd, permit=preparation_permit(cmd)))
    original = resource._set_record

    async def committed_then_raises(digest, stage, receipt, **kwargs):
        await original(digest, stage, receipt, **kwargs)
        if stage == "owned":
            raise OSError("lost acknowledgement after durable commit")

    monkeypatch.setattr(resource, "_set_record", committed_then_raises)
    with pytest.raises(OSError, match="lost acknowledgement"):
        asyncio.run(resource.acquire(cmd))
    lookup = asyncio.run(resource.readback(cmd))
    assert isinstance(lookup, ExactMatch)
    monkeypatch.setattr(resource, "_set_record", original)
    asyncio.run(resource.release(lookup.receipt))
    asyncio.run(store.delete(artifact.id))


def test_uncertainty_diagnostic_failure_does_not_mask_primary_error(tmp_path, monkeypatch):
    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(resource.authorize(cmd, permit=preparation_permit(cmd)))

    primary = OSError("primary read failed")
    diagnostic = RuntimeError("diagnostic write failed")

    async def fail_read(*args, **kwargs):
        raise primary

    async def fail_diagnostic(*args, **kwargs):
        raise diagnostic

    monkeypatch.setattr(store, "read_bytes", fail_read)
    monkeypatch.setattr(resource, "_set_uncertain", fail_diagnostic)
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(resource.acquire(cmd))
    assert error.value.exceptions == (primary, diagnostic)
    assert diagnostic.__context__ is None


@pytest.mark.parametrize("cancel", [True, False])
def test_dispatched_thread_remains_fenced_until_drain(tmp_path, monkeypatch, cancel):
    import cayu.artifacts.local as local
    import cayu.artifacts.resources as resources

    store, artifact = make_store(tmp_path)
    resource_ref = owner("artifact-store")
    resource = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    competitor = LocalArtifactResourceOwner(
        tmp_path / "owner", owner=resource_ref, artifact_store=store
    )
    cmd = command(resource_ref, artifact.id, store=store)
    asyncio.run(competitor.authorize(cmd, permit=preparation_permit(cmd)))
    dispatched, finish = threading.Event(), threading.Event()
    original = local._change_resource_pin
    calls = []

    def blocked(*args):
        original(*args)
        if args[-1]:
            calls.append(args)
            dispatched.set()
            if not finish.wait(10):
                raise AssertionError("test worker was not released")

    monkeypatch.setattr(local, "_change_resource_pin", blocked)
    original_timeout = resources.RESOURCE_FOREGROUND_TIMEOUT_S

    async def run():
        await resource.authorize(cmd, permit=preparation_permit(cmd))
        monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", 5.0 if cancel else 0.1)
        task = asyncio.create_task(resource.acquire(cmd))
        try:
            async with asyncio.timeout(5):
                while not dispatched.is_set():
                    await asyncio.sleep(0.001)
            if cancel:
                task.cancel("first")
                task.cancel("second")
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                else:
                    pytest.fail("caller cancellation was lost")
                assert task.cancelled()
                assert task.cancelling() == 2
            else:
                with pytest.raises(ResourceOwnerUnavailable):
                    await task
                assert not task.cancelled()
            for attempt in (competitor.acquire(cmd), competitor.reconcile()):
                with pytest.raises(ResourceOwnerUnavailable):
                    await attempt
            with pytest.raises(ValueError):
                await store.delete(artifact.id)
            assert len(calls) == 1
        finally:
            monkeypatch.setattr(resources, "RESOURCE_FOREGROUND_TIMEOUT_S", original_timeout)
            finish.set()
            with pytest.raises(ResourceOwnerUnavailable, match="observer stopped"):
                await resource.drain()
        result = await competitor.readback(cmd)
        assert isinstance(result, ExactUnavailable)
        await store.delete(artifact.id)
        assert len(calls) == 1

    asyncio.run(run())
