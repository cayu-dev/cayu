"""Pruning rollback and acknowledgement loss preserve one exact durable cursor."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.producer_pruning_observation import prune_to_receipt
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._request_store import operation_key
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["rollback", "acknowledgement", "cancel", "deadline"])
async def test_public_producer_pruning_failure_replays_exact_batch(
    native_stores, monkeypatch, failure
):
    (
        app,
        resolver,
        admission,
        provider,
        _,
        initialized,
        registration,
        execution,
    ) = await output_scenario(native_stores, with_exports=True)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(registration, execution, context=resolver.recipient.context)
    snapshot = await app.inspect_collaboration_request(
        admission.expected, context=resolver.sender.context
    )
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-for-pruning-failure"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    store = native_stores[0]
    _, rotated = await rotate(store, initialized)
    await app.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-for-pruning-failure"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    key = operation_key(admission.expected.operation)
    for index in range(32):
        state = await app.inspect_collaboration_namespace(context=CONTEXT)
        batch = NamespacePrune(
            operation=rotated.successor.reference.operation(f"failure-prune-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=state.retention_revision,
            max_records=2,
        )
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            cursor = await tx.get("request_pruning", key)
            retained = await tx.get("requests", key)
        if (
            cursor is not None
            and cursor["next_event_index"] == len(retained["event_sequences"])
            and cursor["next_producer_index"] < len(cursor["producer_items"])
        ):
            break
        await app.prune_collaboration_namespace(batch, context=CONTEXT)
    else:
        pytest.fail("Producer pruning never reached its durable inventory phase")

    transaction = store._transaction
    observed = []
    entered, release = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def inject(scope, *, write):
        mutated = False
        async with transaction(scope, write=write) as tx:
            delete = tx.delete

            async def deleting(family, record_key):
                nonlocal mutated
                raw = await tx.get(family, record_key)
                await delete(family, record_key)
                if (
                    family == "operations"
                    and isinstance(raw, dict)
                    and raw.get("mode", "").startswith("producer_")
                ):
                    mutated = True
                    if failure == "rollback" and not observed:
                        observed.append(True)
                        raise RuntimeError("Injected failure after producer record deletion")
                    if failure in {"cancel", "deadline"} and not observed:
                        observed.append(True)
                        entered.set()
                        await release.wait()

            tx.delete = deleting
            yield tx
        if mutated and failure == "acknowledgement" and not observed:
            observed.append(True)
            raise ConnectionError("Producer pruning commit acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_transaction", inject)
        if failure == "deadline":
            # The request coordinator shares this store owner; setup gave it a
            # 60-second functional allowance. Exercise the real 10-second
            # observation deadline explicitly, without changing mutation work.
            patch.setattr(store._owners, "observation_timeout", 10)
            pending_observed = asyncio.Event()
            owner = asyncio.create_task(
                prune_to_receipt(app, batch, pending_observed=pending_observed)
            )
            try:
                await asyncio.wait_for(entered.wait(), 60)
                await asyncio.wait_for(pending_observed.wait(), 20)
                assert not owner.done()
                assert len(store._owners.pending) == 1
                assert not next(iter(store._owners.pending)).done()
                release.set()
                recovered = await asyncio.wait_for(owner, 60)
                assert recovered.removed_records == 2
            finally:
                release.set()
                await asyncio.gather(owner, return_exceptions=True)
        elif failure == "cancel":
            cancellations = []

            async def observer():
                try:
                    return await app.prune_collaboration_namespace(batch, context=CONTEXT)
                except asyncio.CancelledError:
                    cancellations.append(True)
                    raise

            owner = asyncio.create_task(observer())
            contender = None
            try:
                await asyncio.wait_for(entered.wait(), 60)
                owner.cancel()
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 2
                assert cancellations == [True]
                assert len(store._owners.pending) == 1
                contender = asyncio.create_task(
                    app.prune_collaboration_namespace(batch, context=CONTEXT)
                )
                await asyncio.sleep(0)
                assert not contender.done()
                assert len(store._owners.pending) == 1
                release.set()
                recovered = await asyncio.wait_for(contender, 60)
                assert recovered.removed_records == 2
            finally:
                release.set()
                await asyncio.gather(
                    owner, *((contender,) if contender is not None else ()), return_exceptions=True
                )
        else:
            with pytest.raises(CollaborationUnavailable):
                await app.prune_collaboration_namespace(batch, context=CONTEXT)
    assert observed == [True]
    after = await app.inspect_collaboration_namespace(context=CONTEXT)
    if failure == "rollback":
        assert after == state
        async with transaction(initialized.owner.application_scope, write=False) as tx:
            assert await tx.get("request_pruning", key) == cursor
            assert await tx.get("requests", key) == retained
            for item in cursor["producer_items"][cursor["next_producer_index"] :]:
                operation = admission.expected.operation.model_copy(
                    update={"caller_key": item["caller_key"]}
                )
                assert await tx.get("operations", operation_key(operation)) is not None
    else:
        assert after.retention_revision == state.retention_revision + 1
    result = await app.prune_collaboration_namespace(batch, context=CONTEXT)
    assert result.removed_records == 2
    assert await app.prune_collaboration_namespace(batch, context=CONTEXT) == result
    final = await app.inspect_collaboration_namespace(context=CONTEXT)
    assert final.retention_revision == state.retention_revision + 1
    assert not provider.requests
    await app.drain_collaboration_requests()
