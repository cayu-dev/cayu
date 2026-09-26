"""A real cancelled cleanup owner cannot resurrect native history after reclamation."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
async def test_public_cancelled_cleanup_cannot_publish_after_native_retirement(
    native_stores, monkeypatch
):
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, _, initialized, registration, _ = values
    snapshot = await app.inspect_collaboration_request(
        admission.expected, context=resolver.sender.context
    )
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-reclamation-race"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    receiver = app._request_coordinator._registration.receiving_owner
    complete = receiver._complete_producer_cleanup
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(record, *, authority):
        # This is the actual registered receiver entrance with an authority minted
        # by the runtime after source acceptance, not a fabricated test receipt.
        entered.set()
        await release.wait()
        return await complete(record, authority=authority)

    monkeypatch.setattr(receiver, "_complete_producer_cleanup", delayed)
    cancellations = []

    async def caller():
        try:
            return await app.settle_producer_output(registration, context=CONTEXT)
        except asyncio.CancelledError:
            cancellations.append(True)
            raise

    task = asyncio.create_task(caller())
    other = None
    pending = ()
    try:
        await asyncio.wait_for(entered.wait(), 60)
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 2 and cancellations == [True]
        pending = tuple(app._request_coordinator._owners.pending)
        assert pending and not any(item.done() for item in pending)
        store = native_stores[2]()
        other = make_app(
            store,
            app._participant_coordinator._registration,
            session_store=native_stores[1],
            collaboration_requests=RequestRegistration(
                mandates=resolver,
                prepared_admission=PreparedAdmissionRegistration(receiver=registration.receiver),
                max_ttl_ms=300_000,
            ),
        )
        assert await other.initialize_collaboration() == initialized
        monkeypatch.setattr(other._request_coordinator._owners, "observation_timeout", 60)
        assert not other._providers
        await other.settle_producer_output(registration, context=CONTEXT)
        _, rotated = await rotate(store, initialized)
        await other.retire_collaboration_namespace(
            NamespaceRetire(
                operation=rotated.successor.reference.operation("retire-after-duplicate-cleanup"),
                namespace=rotated.namespace.reference,
                expected_revision=rotated.namespace.revision,
                expected_retired_through=0,
            ),
            context=CONTEXT,
        )
        for index in range(16):
            state = await other.inspect_collaboration_namespace(context=CONTEXT)
            receipt = await other.prune_collaboration_namespace(
                NamespacePrune(
                    operation=rotated.successor.reference.operation(f"prune-late-cleanup-{index}"),
                    namespace=rotated.namespace.reference,
                    expected_retention_revision=state.retention_revision,
                    max_records=32,
                ),
                context=CONTEXT,
            )
            if receipt.complete:
                break
        else:
            pytest.fail("Source history did not reach its retirement frontier")
        reclaimed = await other.reclaim_producer_cleanup(
            rotated.namespace.reference, context=CONTEXT
        )
        assert reclaimed.removed == 1 and not reclaimed.remaining
        assert not any(item.done() for item in pending)
        release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 60)
        assert all(isinstance(item, Exception) for item in outcomes)
        assert all(not isinstance(item, asyncio.CancelledError) for item in outcomes)
        assert (
            await other.reclaim_producer_cleanup(rotated.namespace.reference, context=CONTEXT)
        ).removed == 0
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            assert not await tx.scan_operations(initialized.namespace_incarnation, 1, limit=1)
        assert len(provider.requests) == 1
    finally:
        release.set()
        await asyncio.gather(task, *pending, return_exceptions=True)
        await app.drain_collaboration_requests()
        if other is not None:
            await other.drain_collaboration_requests()
