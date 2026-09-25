"""Concrete mandate registration is fixed before acquisition produces output."""

import asyncio

import pytest
from tests.artifacts.test_resources import make_store, preparation_permit, registered_resource

from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    MandateResourcePreparationReader,
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
    ResourceTransferTemplate,
)
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration.mandates import MandateChain, MandateDenied
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.fixture
def artifact_store(tmp_path):
    return make_store(tmp_path)


async def registered_template(
    tmp_path,
    store,
    artifact,
    *,
    planning_deadline=None,
    collaboration_store=None,
    scope=None,
    participant_reference=None,
):
    (
        source,
        command,
        permit,
        source_resolver,
        source_ledger,
        source_initial,
        source_participant,
    ) = await registered_resource(
        tmp_path / "source",
        store,
        artifact,
        collaboration_store,
        bounds=None if planning_deadline is None else {"deadline_at_ms": planning_deadline},
        configuration_revision=None if planning_deadline is None else 1,
        scope=scope,
        participant_reference=participant_reference,
    )
    if collaboration_store is None:
        destination, _, _, resolver, ledger, initialized, participant = await registered_resource(
            tmp_path / "destination", store, artifact, scope=source.owner.application_scope
        )
    else:
        destination, resolver, ledger, initialized, participant = (
            source,
            source_resolver,
            source_ledger,
            source_initial,
            source_participant,
        )
    template = ResourceTransferTemplate(
        operation=initialized.operation("planned-transfer"),
        source=source.owner,
        destination=destination.owner,
        initiator=command.initiator,
        acquisition=command,
        acceptance_generation=1,
    )
    raw = preparation_permit(template)
    transfer_permit = raw.model_copy(
        update={
            "intent": raw.intent.model_copy(
                update={
                    "limits": initialized.binding.limits,
                    "request": raw.intent.request.model_copy(
                        update={
                            "participant": participant,
                            "required_settlement": "quiescence",
                            "expected_configuration_revision": None
                            if planning_deadline is None
                            else 1,
                        }
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
                        update={"resources": (command.intent.selector,)}
                    ),
                )
            )
        }
    )
    registration = destination._preparation_reader
    receiver = MandateResourcePreparationReader(
        owner=registration._owner,
        resolver=resolver,
        context=registration._context,
        redactor=SecretRedactor(),
        registration=registration._registration,
        policy=command.intent.policy,
        artifact_store=store,
        collaboration_store=ledger,
        initialized=initialized,
        responsibilities=() if collaboration_store is None else ((command, permit),),
        transfer_templates=((template, transfer_permit),),
    )
    destination = LocalArtifactResourceOwner(
        destination._journal.root,
        owner=destination.owner,
        artifact_store=store,
        preparation_reader=receiver,
    )
    if collaboration_store is not None:
        source = destination
    return source, command, permit, destination, template, receiver, resolver, source_ledger, ledger


@pytest.mark.parametrize(
    "case", ["success", "revoked", "forged_receipt", "changed_initiator", "changed_permit"]
)
async def test_registered_template_authenticates_actual_source_before_destination_effect(
    tmp_path, monkeypatch, case, artifact_store
):
    store, artifact = artifact_store
    (
        source,
        command,
        permit,
        destination,
        template,
        receiver,
        resolver,
        source_ledger,
        ledger,
    ) = await registered_template(tmp_path, store, artifact)
    frozen_registration = receiver._authority_json()
    try:
        receipt = await source.acquire(
            command, preparation=await source.authorize(command, permit=permit)
        )
        assert receiver._authority_json() == frozen_registration
        transfer = template.bind(receipt)
        expected_permit = receiver.transfer_permit(transfer)
        pins = []
        original_pin = store._pin_resource

        async def counted_pin(*args, **kwargs):
            pins.append(kwargs["owner"])
            return await original_pin(*args, **kwargs)

        monkeypatch.setattr(store, "_pin_resource", counted_pin)
        before = destination._journal.path.read_bytes()
        if case == "revoked":
            resolver.revoked = True
        elif case == "forged_receipt":
            transfer = template.bind(receipt.model_copy(update={"receipt_id": "forged"}))
        elif case == "changed_initiator":
            transfer = transfer.model_copy(
                update={"initiator": transfer.initiator.model_copy(update={"principal": "other"})}
            )
        elif case == "changed_permit":
            expected_permit = expected_permit.model_copy(
                update={
                    "intent": expected_permit.intent.model_copy(
                        update={
                            "request": expected_permit.intent.request.model_copy(
                                update={"expected_lifecycle_revision": 2}
                            )
                        }
                    )
                }
            )
        if case != "success":
            with pytest.raises((ResourceOwnerConflict, ResourceOwnerUnavailable, MandateDenied)):
                await destination.accept_transfer(
                    transfer, source_owner=source, expected_permit=expected_permit
                )
            assert pins == []
            assert destination._journal.path.read_bytes() == before
            return
        accepted = await destination.accept_transfer(
            transfer, source_owner=source, expected_permit=expected_permit
        )
        assert accepted.stage == "accepted" and len(pins) == 1
        await source.release_transferred_source(accepted, destination_owner=destination)
        # Same fixed registration and durable journal recover after the original
        # source pin is gone; replay must not require another acquisition.
        await destination.drain()
        destination = LocalArtifactResourceOwner(
            destination._journal.root,
            owner=destination.owner,
            artifact_store=store,
            preparation_reader=receiver,
        )
        found = await destination.read_transfer(transfer)
        assert isinstance(found, ExactMatch) and found.receipt == accepted
        assert await destination.accept_transfer(transfer, source_owner=source) == accepted
        assert len(pins) == 1
        assert receiver._authority_json() == frozen_registration
        await destination.release_transfer(accepted)
        await store.delete(artifact.id)
    finally:
        await source.drain()
        await destination.drain()
        await source_ledger.close()
        await ledger.close()


async def test_template_cancelled_after_pin_retains_responsibility_until_cleanup(
    tmp_path, monkeypatch, artifact_store
):
    store, artifact = artifact_store
    (
        source,
        command,
        permit,
        destination,
        template,
        receiver,
        resolver,
        source_ledger,
        ledger,
    ) = await registered_template(tmp_path, store, artifact)
    entered, finish = asyncio.Event(), asyncio.Event()
    task = None
    original_pin = store._pin_resource
    calls = []

    async def hold_pin(*args, **kwargs):
        await original_pin(*args, **kwargs)
        calls.append(kwargs["owner"])
        entered.set()
        await finish.wait()

    try:
        receipt = await source.acquire(
            command, preparation=await source.authorize(command, permit=permit)
        )
        transfer = template.bind(receipt)
        monkeypatch.setattr(store, "_pin_resource", hold_pin)
        task = asyncio.create_task(destination.accept_transfer(transfer, source_owner=source))
        async with asyncio.timeout(10):
            await entered.wait()
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 2
        resolver.revoked = True
        with pytest.raises(ResourceOwnerUnavailable):
            await source.release(receipt)
        participant = receiver._responsibilities[0][1].intent.request.participant
        _, pending = await ledger.scan_obligations(
            receiver._initialized,
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
        destination = LocalArtifactResourceOwner(
            destination._journal.root,
            owner=destination.owner,
            artifact_store=store,
            preparation_reader=receiver,
        )
        await destination.reconcile(source_owners=(source,))
        await source.release(receipt)
        await store.delete(artifact.id)
        assert len(calls) == 1
        _, pending = await ledger.scan_obligations(
            receiver._initialized,
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
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await source.drain()
        await destination.drain()
        await source_ledger.close()
        await ledger.close()
