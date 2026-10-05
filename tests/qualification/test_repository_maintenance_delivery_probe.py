"""Negative delivery probes must not collide across a shared-store campaign."""

import asyncio

import pytest

from cayu.delivery.git import (
    RemoteGitDeliveryReconstructionRequiredError,
    RemoteGitDeliveryRepository,
)
from tests.core.test_remote_git_delivery import _coding_publication, _remote_fixture, _request
from tests.qualification.repository_maintenance_delivery_case import changed_destination_request


def test_destination_probes_reopen_independently_without_weakening_conflicts(tmp_path):
    _remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n")

    async def scenario():
        _workspace, publication, _coding_repository, store = await _coding_publication(
            tmp_path, source
        )
        first = _request(publication, base)
        second = first.model_copy(
            update={
                "delivery_id": "delivery-2",
                "idempotency_key": "application-delivery-2",
                "session_id": "second-session",
            }
        )
        probes = tuple(changed_destination_request(request) for request in (first, second))
        assert probes[0].delivery_id != probes[1].delivery_id
        assert probes[0].idempotency_key != probes[1].idempotency_key
        repository = RemoteGitDeliveryRepository(store)
        references = []
        for original, probe in zip((first, second), probes, strict=True):
            assert probe.source == original.source
            assert probe.session_id == original.session_id
            assert probe.repository.expected_base_commit == original.repository.expected_base_commit
            assert probe.repository.destination_ref != original.repository.destination_ref
            references.append(await repository.ensure_request(probe))
        reopened = RemoteGitDeliveryRepository(store)
        for probe, reference in zip(probes, references, strict=True):
            assert await reopened.ensure_request(probe) == reference
        conflict = probes[0].model_copy(update={"session_id": "conflicting-session"})
        with pytest.raises(RemoteGitDeliveryReconstructionRequiredError):
            await reopened.ensure_request(conflict)
        assert await reopened.ensure_request(probes[0]) == references[0]

    asyncio.run(scenario())
