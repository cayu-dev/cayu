"""A committed destination exclusion outlives its bounded host observer."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerMaintenanceRule,
    _ProducerSource,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._producer_delivery_recovery import ProducerDeliveryRecovery
from cayu.collaboration._producer_destination_exclusion import exclusion_operation
from cayu.collaboration._request_store import operation_key

pytestmark = pytest.mark.anyio


async def test_host_exclusion_retains_committed_native_work(native_stores, monkeypatch):
    application, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores, with_exports=True, request_ttl_ms=900_000
    )
    owners = application._request_coordinator._owners
    monkeypatch.setattr(owners, "observation_timeout", 60)
    await application.register_producer_output(
        command, execution, context=resolver.recipient.context
    )
    participant = command.admission.prepared.recipient
    page = await application.pending_producer_outputs(participant, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    destination = command.destinations[0].operation
    key = operation_key(exclusion_operation(command.destinations[0], application._secret_redactor))
    store = native_stores[0]
    transaction = store._transaction
    entered, release = asyncio.Event(), asyncio.Event()
    commits = 0

    @asynccontextmanager
    async def blocked_ack(scope, *, write):
        nonlocal commits
        published = False
        async with transaction(scope, write=write) as tx:
            yield tx
            if write and not commits:
                published = await tx.get("operations", key) is not None
        if published:
            commits += 1
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_transaction", blocked_ack)
    from cayu.collaboration import _host_producer_maintenance

    reconcile = _host_producer_maintenance.reconcile_producer_delivery

    async def short_observation(*args, **kwargs):
        previous = owners.observation_timeout
        owners.observation_timeout = 0.02
        try:
            return await reconcile(*args, **kwargs)
        finally:
            owners.observation_timeout = previous

    monkeypatch.setattr(
        _host_producer_maintenance, "reconcile_producer_delivery", short_observation
    )
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(participant, CONTEXT),),
            producer_rules=(
                _ProducerMaintenanceRule(
                    HostProducerMaintenance(
                        recovery=token, action="exclude", destination=destination
                    ),
                    CONTEXT,
                ),
            ),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01,
        ),
    )
    running = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 90)
        running.cancel()
        running.cancel()
        assert running.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelled()
        await asyncio.sleep(0.05)
        state = await host.aclose()
        assert state.uncertain == 1 and state.failed == 0
        assert not provider.requests
    finally:
        release.set()
        async with asyncio.timeout(90):
            while (await host.aclose()).pending:
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                await asyncio.sleep(0.01)
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
    expected = ProducerDeliveryRecovery(
        registration=token.registration,
        registration_commitment=token.registration_commitment,
        destination=destination,
    )
    receipt = await application.reconcile_producer_delivery(expected, context=CONTEXT)
    assert receipt.state == "excluded"
    assert (
        await application.reconcile_producer_delivery(expected, context=CONTEXT, exclude=True)
        == receipt
    )
    assert commits == 1
    assert not provider.requests
    assert host.inspect().uncertain == 0
