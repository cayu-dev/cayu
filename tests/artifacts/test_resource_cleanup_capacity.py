"""Fixed native registrations reserve exclusion before optional work can fill it."""

import json

import pytest
from tests.artifacts.test_resource_transfer_templates import registered_template
from tests.artifacts.test_resources import make_store

from cayu.artifacts import resources
from cayu.artifacts._resource_cleanup_capacity import (
    bind_cleanup_capacity,
    reservations,
)
from cayu.artifacts.resources import (
    LocalArtifactResourceOwner,
    ResourceOwnerConflict,
    ResourceOwnerError,
    ResourceOwnerUnavailable,
)
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.fixture
def artifact_store(tmp_path):
    return make_store(tmp_path)


@pytest.mark.parametrize("dimension", ["events", "bytes", "nodes", "operations"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
async def test_registration_cleanup_envelope_below_at_above_limit(
    tmp_path, artifact_store, monkeypatch, dimension, offset
):
    store, artifact = artifact_store
    ledger = InMemoryCollaborationStore()
    owner = None
    try:
        owner, _, _, _, _, reader, _, _, _ = await registered_template(
            tmp_path, store, artifact, collaboration_store=ledger
        )
        with owner._journal.locked() as journal:
            assert len(reservations(journal)) == 2
            assert resources._event_usage(journal) == 4
            envelope = resources._journal_capacity_envelope(journal)
        if dimension == "events":
            required = 4
            name = "RESOURCE_MAX_EVENTS"
        elif dimension == "operations":
            # One acquisition and one transfer reserve their distinct families.
            required = 1
            name = "RESOURCE_MAX_OPERATIONS"
        elif dimension == "bytes":
            required = len(
                json.dumps(
                    envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            )
            name = "RESOURCE_JOURNAL_MAX_BYTES"
        else:

            def nodes(value):
                if isinstance(value, dict):
                    return 1 + sum(nodes(item) for item in value.values())
                if isinstance(value, list):
                    return 1 + sum(nodes(item) for item in value)
                return 1

            required = nodes(envelope)
            name = "RESOURCE_JOURNAL_MAX_NODES"
        monkeypatch.setattr(resources, name, required + offset)
        root = tmp_path / "bounded-registration"

        def construct():
            return LocalArtifactResourceOwner(
                root, owner=owner.owner, artifact_store=store, preparation_reader=reader
            )

        if offset < 0:
            with pytest.raises((ResourceOwnerError, ValueError)):
                construct()
            # Only inert owner identity may have committed. No partial recipe
            # reservation, operation, event or physical resource pin is admitted.
            rejected = json.loads((root / "resource-owner.json").read_text())
            assert rejected["cleanup_reservations"] == {}
        else:
            prepared = construct()
            before = prepared._journal.path.read_bytes()
            reopened = construct()
            assert reopened._journal.path.read_bytes() == before
            with reopened._journal.locked() as retained:
                assert retained["operations"] == retained["transfers"] == {}
                assert retained["events"] == []
                assert resources._event_usage(retained) == 4
        await store.delete(artifact.id)
    finally:
        if owner is not None:
            await owner.drain()
        await ledger.close()


async def test_cleanup_reservation_overlap_and_conflict_are_exact(
    tmp_path, artifact_store, monkeypatch
):
    store, artifact = artifact_store
    ledger = InMemoryCollaborationStore()
    owner = None
    try:
        owner, command, permit, _, template, reader, _, _, _ = await registered_template(
            tmp_path, store, artifact, collaboration_store=ledger
        )
        monkeypatch.setattr(resources, "RESOURCE_MAX_EVENTS", 9)
        acquired = await owner.acquire(
            command, preparation=await owner.authorize(command, permit=permit)
        )
        with owner._journal.locked() as journal:
            assert resources._event_usage(journal) == 9  # 7 live + 2 unused, not 11
        transfer = template.bind(acquired)
        with pytest.raises(ResourceOwnerError, match="event reservation"):
            await owner.accept_transfer(transfer, source_owner=owner)
        before = owner._journal.path.read_bytes()
        wrong = permit.model_copy(
            update={"initiator": permit.initiator.model_copy(update={"principal": "different"})}
        )
        with pytest.raises(ResourceOwnerConflict), owner._journal.locked() as journal:
            bind_cleanup_capacity(journal, ((command, wrong),), redactor=SecretRedactor())
        assert owner._journal.path.read_bytes() == before
        excluded = await owner.settle_preparation(
            transfer, permit=reader.transfer_permit(transfer), source_owner=owner
        )
        assert excluded.proves_exclusion
        with owner._journal.locked() as journal:
            # The unused two-slot reservation became its two actual events;
            # it is not counted twice and source cleanup remains reserved.
            assert resources._event_usage(journal) == 9
        monkeypatch.setattr(resources, "RESOURCE_MAX_EVENTS", 100)
        # More room cannot authorize a delayed transfer after exact exclusion.
        reopened = LocalArtifactResourceOwner(
            owner._journal.root,
            owner=owner.owner,
            artifact_store=store,
            preparation_reader=reader,
        )
        with pytest.raises(ResourceOwnerUnavailable):
            await reopened.accept_transfer(transfer, source_owner=owner)
        assert (
            await reopened.settle_preparation(
                transfer, permit=reader.transfer_permit(transfer), source_owner=owner
            )
            == excluded
        )
        await owner.release(acquired)
        with owner._journal.locked() as journal:
            assert resources._event_usage(journal) == len(journal["events"]) == 7
        await store.delete(artifact.id)
    finally:
        if owner is not None:
            await owner.drain()
        await ledger.close()
