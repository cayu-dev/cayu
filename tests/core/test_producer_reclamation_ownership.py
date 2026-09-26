"""Public reclamation retains its owner across cancellation and lost acknowledgement."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl


async def retired_producer(native_stores, monkeypatch):
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, _, initialized, registration, _ = values
    snapshot = await app.inspect_collaboration_request(
        admission.expected, context=resolver.sender.context
    )
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-before-native-reclamation"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    await app.settle_producer_output(registration, context=CONTEXT)
    _, rotated = await rotate(native_stores[0], initialized)
    await app.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-before-native-reclamation"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    for index in range(16):
        state = await app.inspect_collaboration_namespace(context=CONTEXT)
        receipt = await app.prune_collaboration_namespace(
            NamespacePrune(
                operation=rotated.successor.reference.operation(
                    f"reclamation-prerequisite-{index}"
                ),
                namespace=rotated.namespace.reference,
                expected_retention_revision=state.retention_revision,
                max_records=32,
            ),
            context=CONTEXT,
        )
        if receipt.complete:
            return app, provider, rotated.namespace.reference
    pytest.fail("Source history did not reach its retirement frontier")


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["acknowledgement", "cancel-before", "cancel-after"])
async def test_public_reclamation_preserves_committed_fence_and_owned_completion(
    native_stores, monkeypatch, failure
):
    app, provider, namespace = await retired_producer(native_stores, monkeypatch)
    receiver = app._request_coordinator._registration.receiving_owner
    retire = receiver._retire_producer_cleanup
    entered, release = asyncio.Event(), asyncio.Event()
    results = []

    async def intercept(retirement, *, authority, limit):
        # The real receiver receives the runtime's sealed authority; the native
        # transaction itself is never replaced by a fabricated success receipt.
        if failure == "cancel-before":
            entered.set()
            await release.wait()
        result = await retire(retirement, authority=authority, limit=limit)
        results.append(result)
        if failure == "acknowledgement":
            raise ConnectionError("Native retirement acknowledgement lost")
        if failure == "cancel-after":
            entered.set()
            await release.wait()
        return result

    cancellations = []

    async def observer():
        try:
            return await app.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=1)
        except asyncio.CancelledError:
            cancellations.append(True)
            raise

    pending = ()
    task = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(receiver, "_retire_producer_cleanup", intercept)
            if failure == "acknowledgement":
                with pytest.raises(CollaborationUnavailable):
                    await observer()
            else:
                task = asyncio.create_task(observer())
                await asyncio.wait_for(entered.wait(), 60)
                task.cancel()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 2
                assert cancellations == [True]
                pending = tuple(app._request_coordinator._owners.pending)
                assert pending and not any(item.done() for item in pending)
                assert len(results) == int(failure == "cancel-after")
                release.set()
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), 60
                )
                assert not any(isinstance(item, BaseException) for item in outcomes)
        assert len(results) == 1
        assert results[0].removed == 1 and not results[0].remaining
        # This is a state-based drain, not an exact per-call receipt replay. A
        # lost observer cannot reopen the retired namespace or re-remove its ACK.
        reconciled = await app.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=1)
        assert reconciled.removed == 0 and not reconciled.remaining
        assert len(provider.requests) == 1
    finally:
        release.set()
        await asyncio.gather(
            *pending, *((task,) if task is not None else ()), return_exceptions=True
        )
        await app.drain_collaboration_requests()
