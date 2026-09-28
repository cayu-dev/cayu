"""Exact native outcomes release failed host turns without replaying production."""

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


@pytest.mark.parametrize("action", ["export", "publish_answer", "deliver"])
async def test_committed_maintenance_ack_loss_releases_exact_local_turn(
    native_stores, monkeypatch, action, cancelled_readback=False, later_readback=False
):
    from cayu.collaboration import _host_producer_maintenance as adapter

    values, access = await completed_export_scenario(native_stores, monkeypatch)
    app, _, _, provider, _, _, command, _ = values
    participant = command.admission.prepared.recipient
    page = await app.pending_producer_outputs(participant, context=CONTEXT)
    recovery = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    destination = command.destinations[0].operation
    if action != "export":
        exported = await app.export_producer_output(command, destination, context=access)
    if action == "deliver":
        from cayu.collaboration._session_export_store import digest

        await app.publish_producer_outcome(command, destination=destination, context=access)
        source = await app.lookup_session_export(exported.request, context=access)
        policy = app._session_export_coordinator.registration.policy
        policy.register_export(
            source.receipt,
            payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
            consumer_id=command.destinations[0].recipient.participant_id,
        )
        policy.allowed_receipts.add(command.operation.caller_key)
    name = {
        "export": "export_producer_output",
        "publish_answer": "publish_producer_outcome",
        "deliver": "deliver_producer_output",
    }[action]
    operation = getattr(adapter, name)
    primary = ExceptionGroup(
        "lost acknowledgement", [ConnectionError("transport"), OSError("close")]
    )
    receipts = []

    async def commit_then_fail(*args, **kwargs):
        receipts.append(await operation(*args, **kwargs))
        raise primary

    monkeypatch.setattr(adapter, name, commit_then_fail)
    entered, release = asyncio.Event(), asyncio.Event()
    readbacks = []
    if cancelled_readback:
        from cayu.collaboration import _host_producer_reconciliation

        inspect = _host_producer_reconciliation.inspect_producer_output

        async def blocked_readback(*args, **kwargs):
            from cayu.collaboration._contracts import ExactUnavailable

            readbacks.append(True)
            if later_readback and len(readbacks) == 1:
                return ExactUnavailable()
            found = await inspect(*args, **kwargs)
            entered.set()
            await release.wait()
            return found

        monkeypatch.setattr(
            _host_producer_reconciliation, "inspect_producer_output", blocked_readback
        )
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(HostProducerSource(participant, CONTEXT),),
            producer_rules=(
                HostProducerMaintenanceRule(
                    HostProducerMaintenance(
                        recovery=recovery, action=action, destination=destination
                    ),
                    CONTEXT,
                    access,
                ),
            ),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01 if cancelled_readback else 30,
        ),
    )
    async with host, asyncio.timeout(180):
        if cancelled_readback:
            observer = asyncio.create_task(host.run())
            try:
                await asyncio.wait_for(entered.wait(), 120)
                observer.cancel()
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 2
                state = await host.aclose()
                assert state.pending and state.uncertain == 1
                assert state.failed == int(later_readback)
                assert len(readbacks) == (2 if later_readback else 1)
            finally:
                release.set()
                if not observer.done():
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
            failures = []
            while True:
                try:
                    if not (await host.aclose()).pending:
                        break
                except ExceptionGroup as error:
                    failures.append(error)
                await asyncio.sleep(0.01)
            assert failures == [primary]
        else:
            with pytest.raises(ExceptionGroup) as caught:
                await host.run()
            assert caught.value is primary
        assert len(receipts) == 1
        assert host.inspect().uncertain == host.inspect().failed == 0
    assert not host.inspect().pending
    snapshot = (await app.inspect_producer_output(recovery, context=CONTEXT)).receipt
    assert snapshot.destinations[0].export == "published"
    if action in ("publish_answer", "deliver"):
        assert snapshot.request_state == "answered"
        assert snapshot.answer_destination == destination
    if action == "deliver":
        assert snapshot.destinations[0].delivery == "appended"
    assert snapshot.cleanup_ack is None  # Local turn release is not producer cleanup.
    assert len(provider.requests) == 1


async def test_maintenance_reconciliation_keeps_ownership_after_observer_cancel(
    native_stores, monkeypatch
):
    await test_committed_maintenance_ack_loss_releases_exact_local_turn(
        native_stores, monkeypatch, "export", cancelled_readback=True
    )


async def test_later_exact_reconciliation_drains_the_original_failed_host(
    native_stores, monkeypatch
):
    await test_committed_maintenance_ack_loss_releases_exact_local_turn(
        native_stores, monkeypatch, "export", cancelled_readback=True, later_readback=True
    )
