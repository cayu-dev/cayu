"""Maintenance fixtures observe exact native owners, never guess settlement."""

import asyncio

import pytest
from tests.core.producer_pruning_observation import retire_to_receipt
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration import _namespace_store
from cayu.collaboration.lifecycle import NamespaceRetire
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["committed", "failed", "ack_lost"])
def test_retirement_observation_requires_exact_owner_receipt(
    tmp_path, monkeypatch, backend, outcome
):
    async def run():
        store = (
            InMemoryCollaborationStore()
            if backend == "memory"
            else SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        )
        application = app(store, registration())
        initialized = await application.initialize_collaboration()
        _, rotated = await rotate(store, initialized)
        request = NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        )
        original = _namespace_store.apply_lifecycle
        entered, release, pending_observed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls, receipts = [], []
        failure = ConnectionError("controlled retirement failure")

        async def held(*args, **kwargs):
            calls.append(args[2])
            entered.set()
            await release.wait()
            if outcome == "failed":
                raise failure
            receipt = await original(*args, **kwargs)
            receipts.append(receipt)
            if outcome == "ack_lost":
                raise failure
            return receipt

        monkeypatch.setattr(_namespace_store, "apply_lifecycle", held)
        original_run = store._owners.run

        async def short_mutation_observation(operation, **kwargs):
            if kwargs["key"][0] != "mutation":
                return await original_run(operation, **kwargs)
            timeout = store._owners.observation_timeout
            store._owners.observation_timeout = 0.01
            try:
                return await original_run(operation, **kwargs)
            finally:
                store._owners.observation_timeout = timeout

        # Expire only the held mutation's observation. Readback and teardown
        # keep their normal bounds and are not additional injected failures.
        monkeypatch.setattr(store._owners, "run", short_mutation_observation)
        observer = asyncio.create_task(
            retire_to_receipt(application, request, pending_observed=pending_observed)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(pending_observed.wait(), 5)
            owners = tuple(store._owners.pending)
            assert len(owners) == 1 and not owners[0].done()
            assert not observer.done()
            release.set()
            if outcome == "committed":
                receipt = await asyncio.wait_for(asyncio.shield(observer), 5)
                assert receipt == receipts[0]
                assert receipt.namespace.state == "retired"
            else:
                with pytest.raises(ConnectionError) as caught:
                    await asyncio.wait_for(asyncio.shield(observer), 5)
                assert caught.value is failure
            if outcome != "failed":
                replay = await application.retire_collaboration_namespace(request, context=CONTEXT)
                assert replay == receipts[0]
            assert len(calls) == 1
            assert not owners[0].cancelled()
            assert not store._owners.pending
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await store.close()

    asyncio.run(run())
