"""Draining one application's collaboration requests leaves a shared store usable."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import public_setup

from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


async def _second_app(store, values, resolver):
    other = identity_tests.app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await other.initialize_collaboration()
    return other


async def test_draining_one_app_does_not_close_the_shared_store(stores):
    store = stores()
    first, resolver, values = await public_setup(store)
    _initialized, sender, _recipient, request = values[1:5]
    second = await _second_app(store, values, resolver)

    await first.drain_collaboration_requests()

    # The draining application refuses new work of its own.
    with pytest.raises(CollaborationUnavailable) as refused:
        await first.accept_collaboration_request(request, context=resolver.context)
    assert "requests are closing" in str(refused.value.__cause__)
    # Another application sharing the store keeps working.
    receipt = await second.accept_collaboration_request(
        request.model_copy(update={"target": sender.reference}), context=resolver.context
    )
    assert receipt.expected.intent.selection.recipient.reference == sender.reference
    assert store._owners.closed is False
    await second.drain_collaboration_requests()


async def test_app_drain_waits_for_its_work_and_the_store_still_admits_it(stores, monkeypatch):
    store = stores()
    first, resolver, values = await public_setup(store)
    _initialized, sender, _recipient, request = values[1:5]
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction
    armed = True

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal armed
        if write and armed:
            armed = False
            entered.set()
            await release.wait()
        async with original(scope, write=write) as tx:
            yield tx

    monkeypatch.setattr(store, "_transaction", transaction)
    caller = asyncio.create_task(
        first.accept_collaboration_request(
            request.model_copy(update={"target": sender.reference}), context=resolver.context
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # The in-flight mutation belongs to this application and to the store.
        scope_pending = first._request_coordinator.owners.pending
        assert scope_pending and scope_pending <= store._owners.pending
        drain = asyncio.create_task(first.drain_collaboration_requests(timeout_s=5))
        await asyncio.sleep(0.01)
        assert not drain.done()
        release.set()
        await asyncio.wait_for(drain, 5)
        # The store admitted the already-started mutation to finish.
        await asyncio.wait_for(caller, 5)
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def test_app_drain_reports_unsettled_work_within_its_timeout(stores, monkeypatch):
    store = stores()
    first, resolver, values = await public_setup(store)
    _initialized, sender, _recipient, request = values[1:5]
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction

    @asynccontextmanager
    async def transaction(scope, *, write):
        if write:
            entered.set()
            await release.wait()
        async with original(scope, write=write) as tx:
            yield tx

    monkeypatch.setattr(store, "_transaction", transaction)
    caller = asyncio.create_task(
        first.accept_collaboration_request(
            request.model_copy(update={"target": sender.reference}), context=resolver.context
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(CollaborationUnavailable, match="still draining"):
            await first.drain_collaboration_requests(timeout_s=0.05)
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def test_app_drain_waits_for_work_run_directly_on_the_store(stores):
    store = stores()
    first, _resolver, _values = await public_setup(store)
    started, release = asyncio.Event(), asyncio.Event()

    async def store_side_mutation() -> None:
        started.set()
        await release.wait()

    # Some application paths (wait registration, permits) run on the store's
    # owners directly; an observer that stops waiting leaves the work owned.
    observer = asyncio.create_task(
        store._owners.run(
            store_side_mutation,
            key=("store-side",),
            expectation=b"store-side",
            redactor=SecretRedactor(),
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    observer.cancel()
    await asyncio.gather(observer, return_exceptions=True)
    assert store._owners.pending
    with pytest.raises(CollaborationUnavailable, match="still draining"):
        await first.drain_collaboration_requests(timeout_s=0.05)
    release.set()
    await first.drain_collaboration_requests(timeout_s=5)
    assert store._owners.closed is False
