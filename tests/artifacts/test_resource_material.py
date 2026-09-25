"""Compact material references remain exact and authorized after native restart."""

import pytest
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.artifacts.test_resources import make_store

from cayu.artifacts._resource_material import ResourceMaterialReference, material_reference
from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    ResourceOwnerConflict,
    ResourceOwnerUnavailable,
)
from cayu.collaboration.mandates import MandateDenied

pytestmark = pytest.mark.anyio


@pytest.fixture
def artifact_store(tmp_path):
    return make_store(tmp_path)


async def test_material_reference_exact_restart_revocation_and_release(tmp_path, artifact_store):
    store, artifact = artifact_store
    (
        source,
        command,
        permit,
        destination,
        template,
        reader,
        resolver,
        source_ledger,
        ledger,
    ) = await registered_template(tmp_path, store, artifact)
    try:
        acquisition = await source.acquire(
            command, preparation=await source.authorize(command, permit=permit)
        )
        transfer = await destination.accept_transfer(
            template.bind(acquisition), source_owner=source
        )
        preparation = await destination.read_preparation(transfer.command)
        reference = material_reference(transfer, preparation)
        assert len(reference.model_dump_json().encode()) < 2048
        expected = await destination.read_material(reference)
        await source.release_transferred_source(transfer, destination_owner=destination)
        await destination.drain()
        destination = LocalArtifactResourceOwner(
            destination._journal.root,
            owner=destination.owner,
            artifact_store=store,
            preparation_reader=reader,
        )
        reference = ResourceMaterialReference.model_validate_json(reference.model_dump_json())
        assert await destination.read_material(reference) == expected
        before = destination._journal.path.read_bytes()
        for field in ("template_commitment", "transfer_commitment", "preparation_commitment"):
            with pytest.raises(ResourceOwnerConflict):
                await destination.read_material(
                    reference.model_copy(update={field: "sha256:" + "0" * 64})
                )
            assert destination._journal.path.read_bytes() == before
        resolver.revoked = True
        with pytest.raises(MandateDenied):
            await destination.read_material(reference)
        assert destination._journal.path.read_bytes() == before
        resolver.revoked = False
        await destination.release_transfer(transfer)
        with pytest.raises(ResourceOwnerUnavailable):
            await destination.read_material(reference)
        await store.delete(artifact.id)
    finally:
        resolver.revoked = False
        await source.drain()
        await destination.drain()
        await source_ledger.close()
        await ledger.close()
