"""A failed source read is not an indefinitely running native effect."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import (
    CollaborationHost,
    HostOwnershipLimits,
    HostProducerMaintenance,
    HostProducerMaintenanceRule,
    HostProducerSource,
    HostRegistration,
)

pytestmark = pytest.mark.anyio


async def test_failed_preflight_releases_only_local_slot_and_preserves_error(
    native_stores, monkeypatch
):
    from cayu.collaboration import _host_producer_maintenance

    values, access = await completed_export_scenario(native_stores, monkeypatch)
    app, _, _, provider, _, _, command, _ = values
    participant = command.admission.prepared.recipient
    page = await app.pending_producer_outputs(participant, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    destination = command.destinations[0].operation
    lookup = _host_producer_maintenance.lookup_producer_registration
    failure = ExceptionGroup("source read failed", [OSError("read"), RuntimeError("read cleanup")])
    reads = []

    async def fail_first_read(*args, **kwargs):
        reads.append(True)
        if len(reads) == 1:
            raise failure
        return await lookup(*args, **kwargs)

    monkeypatch.setattr(_host_producer_maintenance, "lookup_producer_registration", fail_first_read)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(HostProducerSource(participant, CONTEXT),),
            producer_rules=(
                HostProducerMaintenanceRule(
                    HostProducerMaintenance(
                        recovery=token, action="export", destination=destination
                    ),
                    CONTEXT,
                    access,
                ),
            ),
            observation_timeout_s=30,
            shutdown_timeout_s=30,
        ),
    )
    async with host, asyncio.timeout(120):
        with pytest.raises(ExceptionGroup) as caught:
            await host.run()
        assert caught.value is failure
        assert host.inspect().uncertain == host.inspect().failed == 0
        assert reads == [True] and len(provider.requests) == 1
        current = await app.inspect_producer_output(token, context=CONTEXT)
        assert current.receipt.destinations[0].export is None
        assert current.receipt.cleanup_ack is None
        # Explicit servicing retries the original native operation under fresh
        # authorization. The failed read cannot permanently consume its slot.
        while True:
            state = await host.service_once()
            assert state.failed == state.source_failures == 0
            if state.serviced:
                break
            await asyncio.sleep(0.01)
    assert not host.inspect().pending
    current = await app.inspect_producer_output(token, context=CONTEXT)
    assert current.receipt.destinations[0].export == "published"
    assert current.receipt.cleanup_ack is None
    assert len(provider.requests) == 1
