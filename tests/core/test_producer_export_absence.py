"""A recorded source intent needs a positive fence before any native export exists."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._producer_export_store import read_export
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
@pytest.mark.parametrize("revoked", [False, True])
@pytest.mark.parametrize("ack_loss", [False, True])
@pytest.mark.parametrize("exclude_first", [False, True, "concurrent"])
async def test_native_absent_export_exclusion_fences_delayed_first_publication(
    native_stores, monkeypatch, revoked, ack_loss, exclude_first
):
    values, context = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, session, initialized, proposal, _ = values
    destination = proposal.destinations[0]
    exports = app._session_export_coordinator
    native_export = exports.export
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(request, **kwargs):
        entered.set()
        await release.wait()
        return await native_export(request, **kwargs)

    monkeypatch.setattr(exports, "export", delayed)
    observer = asyncio.create_task(
        app.export_producer_output(proposal, destination.operation, context=context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 60)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 1
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=False
        ) as tx:
            current = await retained_request(
                native_stores[0],
                tx,
                initialized,
                admission.expected.intent.request,
                admission.expected.initiator,
                app._secret_redactor,
            )
            intent = await read_export(tx, proposal, destination, redactor=app._secret_redactor)
        assert current is not None and intent is not None
        assert (
            await exports._record(await app.session_store.load(session.id), intent.request) is None
        )
        await app.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-before-native-export"),
                expected=admission.expected,
                expected_revision=current.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
        resolver.recipient.denied = revoked
        with monkeypatch.context() as patch:
            patch.setattr(
                app.session_store, "_supports_producer_attachment_protocol", lambda: False
            )
            with pytest.raises(CollaborationUnavailable):
                await app.retire_producer_export(proposal, destination.operation, context=CONTEXT)
        assert (
            await exports._record(await app.session_store.load(session.id), intent.request) is None
        )
        from cayu.collaboration import _producer_export_absence as absence

        publish = absence.publish_export_mutation
        committed_exclusions = []

        async def publish_then_lose_ack(*args, **kwargs):
            await publish(*args, **kwargs)
            committed_exclusions.append(args[5])
            raise ConnectionError("Native exclusion acknowledgement lost")

        if ack_loss:
            monkeypatch.setattr(absence, "publish_export_mutation", publish_then_lose_ack)
        from cayu import ProducerDeliveryRecovery

        pending = await app.pending_producer_outputs(
            proposal.admission.prepared.recipient, context=CONTEXT
        )
        token = next(
            item.recovery
            for item in pending.items
            if item.recovery.registration == proposal.operation
        )
        recovery = ProducerDeliveryRecovery(**token.model_dump(), destination=destination.operation)
        if exclude_first is True:
            await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
        if exclude_first == "concurrent":
            status, _ = await asyncio.gather(
                app.retire_producer_export(proposal, destination.operation, context=CONTEXT),
                app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True),
            )
        else:
            status = await app.retire_producer_export(
                proposal, destination.operation, context=CONTEXT
            )
        assert status.state == "excluded"
        excluded = await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
        assert excluded.state == "excluded"
        assert await app.reconcile_producer_delivery(recovery, context=CONTEXT) == excluded
        assert (
            await app.retire_producer_export(proposal, destination.operation, context=CONTEXT)
            == status
        )
        backend, address = native_stores[3]
        if backend != "memory":
            from tests.core.test_producer_excluded_export_cleanup import process_readback

            assert await process_readback(backend, address, proposal) == status
        if ack_loss:
            assert len(committed_exclusions) == 1
            assert committed_exclusions[0]["mode"] == "producer_export_exclusion"
        release.set()
        await asyncio.wait_for(
            asyncio.gather(
                *tuple(exports.owners.pending),
                *tuple(app._request_coordinator._owners.pending),
                return_exceptions=True,
            ),
            60,
        )
        assert exports.projectors[destination.projector].calls == 0
        assert (
            await app.retire_producer_export(proposal, destination.operation, context=CONTEXT)
            == status
        )
        root = await exports.root(await app.session_store.load(session.id))
        assert (root.producer_exclusion_count, root.producer_exclusion_bytes) == (1, 65536)
        if ack_loss:
            assert len(committed_exclusions) == 1
        final = await app.settle_producer_output(proposal, context=CONTEXT)
        assert final.delivery == "excluded" and len(provider.requests) == 1
        await app.session_store.delete_session(session.id)
        assert await app.settle_producer_output(proposal, context=CONTEXT) == final
        assert (
            await app.retire_producer_export(proposal, destination.operation, context=CONTEXT)
            == status
        )
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await exports.owners.drain()
        await app._request_coordinator._owners.drain()
