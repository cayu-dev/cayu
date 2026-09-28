"""Host expiry traverses current authority and the real request-control owner."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_participant_identity import stores as stores

from cayu.collaboration._host import CollaborationHost, _HostRegistration
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_requests import _RequestMaintenanceSource

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_request_expiry_uses_owner_clock_after_reopen(stores, monkeypatch, cancel_observer):
    original_app, resolver, values = await public_setup(stores())
    accepted = await original_app.accept_collaboration_request(values[4], context=resolver.context)
    # The replacement has no process-local acceptance handle and no provider.
    store = stores()
    application = app(
        store,
        original_app._participant_coordinator._registration,
        collaboration_requests=original_app._request_coordinator._registration,
    )
    await application.initialize_collaboration()
    deadline = accepted.expected.intent.selection.expires_at_ms
    now = deadline - 1
    original_transaction = store._transaction
    entered, release = asyncio.Event(), asyncio.Event()
    blocked = False

    @asynccontextmanager
    async def at_owner_time(scope, *, write):
        nonlocal blocked
        async with original_transaction(scope, write=write) as tx:

            async def now_ms():
                return now

            tx.now_ms = now_ms
            yield tx
        if cancel_observer and write and not blocked:
            blocked = True
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_transaction", at_owner_time)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            request_maintenance_sources=(_RequestMaintenanceSource(resolver.context),),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01 if cancel_observer else 30,
        ),
    )
    await host.service_once()
    async with asyncio.timeout(30):
        while host.inspect().discovery_pending:
            await host.service_once()
    assert not host.inspect().failed
    assert not host.inspect().source_failures
    assert host.inspect().uncertain == 0
    before = await application.inspect_collaboration_request(
        accepted.expected, context=resolver.context
    )
    assert before.state == "open" and before.revision == 1
    now = deadline
    # A prior authenticated due scan does not authorize a later expiry. Restore
    # current authority explicitly; no expiry may be dispatched while revoked.
    resolver.denied = True
    async with asyncio.timeout(30):
        while not host.inspect().source_failures:
            await host.service_once()
    assert host.inspect().source_failures == 1
    assert host.inspect().uncertain == 0
    resolver.denied = False
    unchanged = await application.inspect_collaboration_request(
        accepted.expected, context=resolver.context
    )
    assert unchanged == before
    running = None
    try:
        if cancel_observer:
            running = asyncio.create_task(host.run())
            await asyncio.wait_for(entered.wait(), 30)
            running.cancel()
            running.cancel()
            assert running.cancelling() == 2
            with pytest.raises(asyncio.CancelledError):
                await running
            assert running.cancelled()
            closed = await host.aclose()
            assert closed.uncertain == 1 and closed.failed == 0
            # The source has committed expiry, but the late ACK still belongs
            # to the host; absence from due discovery cannot release that slot.
            page = await application.list_due_collaboration_requests(context=resolver.context)
            assert page.items == ()
        else:
            async with asyncio.timeout(60):
                while True:
                    observed = await host.service_once()
                    for outcome in host._owned.inspect().completed:
                        if outcome.error is not None:
                            raise outcome.error
                    if observed.serviced:
                        assert observed.source_failures == 0, host._request_maintenance.errors
                        break
    finally:
        release.set()
        async with asyncio.timeout(30):
            while (await host.aclose()).pending:
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                await asyncio.sleep(0.01)
        if running is not None and not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        try:
            after = await application.inspect_collaboration_request(
                accepted.expected, context=resolver.context
            )
            participant = await application.inspect_participant(
                values[3].reference, context=CONTEXT
            )
        finally:
            await application.drain_collaboration_requests()
            await original_app.drain_collaboration_requests()
    assert after.state == "expired" and after.revision == 2
    assert after.terminal.elected_at_ms == deadline
    assert host.inspect().uncertain == 0
    assert participant.outstanding_obligations == 0


async def test_independent_hosts_expire_the_same_observed_revision(stores, monkeypatch):
    first, resolver, values = await public_setup(stores())
    accepted = await first.accept_collaboration_request(values[4], context=resolver.context)
    second = app(
        stores(),
        first._participant_coordinator._registration,
        collaboration_requests=first._request_coordinator._registration,
    )
    await second.initialize_collaboration()
    deadline = accepted.expected.intent.selection.expires_at_ms
    patched = set()
    for application in (first, second):
        store, _ = application._participant_coordinator._ready()
        if id(store) in patched:
            continue
        patched.add(id(store))

        def owner_clock(transaction):
            @asynccontextmanager
            async def at_deadline(scope, *, write):
                async with transaction(scope, write=write) as tx:

                    async def now_ms():
                        return deadline

                    tx.now_ms = now_ms
                    yield tx

            return at_deadline

        monkeypatch.setattr(store, "_transaction", owner_clock(store._transaction))
    observed = 0
    both_observed = asyncio.Event()

    def together(discover):
        first_read = True

        async def read(**kwargs):
            nonlocal observed, first_read
            page = await discover(**kwargs)
            if first_read:
                first_read = False
                assert len(page.items) == 1 and page.items[0].revision == 1
                observed += 1
                if observed == 2:
                    both_observed.set()
                await both_observed.wait()
            return page

        return read

    hosts = []
    for application in (first, second):
        monkeypatch.setattr(
            application._request_coordinator,
            "due",
            together(application._request_coordinator.due),
        )
        hosts.append(
            CollaborationHost(
                application,
                _HostRegistration(
                    limits=HostOwnershipLimits(1, 1, 2, 262144),
                    producer_sources=(),
                    producer_rules=(),
                    request_maintenance_sources=(_RequestMaintenanceSource(resolver.context),),
                    observation_timeout_s=30,
                    shutdown_timeout_s=30,
                ),
            )
        )
    settled = set()
    try:
        async with asyncio.timeout(60):
            while len(settled) < 2:
                outcomes = await asyncio.gather(*(host.service_once() for host in hosts))
                for index, (host, outcome) in enumerate(zip(hosts, outcomes, strict=True)):
                    for item in host._owned.inspect().completed:
                        if item.error is not None:
                            raise item.error
                    assert outcome.source_failures == 0, host._request_maintenance.errors
                    if outcome.serviced:
                        settled.add(index)
        retained = await first.inspect_collaboration_request(
            accepted.expected, context=resolver.context
        )
        assert retained.state == "expired" and retained.revision == 2
        assert (
            await second.inspect_collaboration_request(accepted.expected, context=resolver.context)
            == retained
        )
        participant = await first.inspect_participant(values[3].reference, context=CONTEXT)
        assert participant.outstanding_obligations == 0
    finally:
        both_observed.set()
        for host in hosts:
            assert (await host.aclose()).uncertain == 0
        await first.drain_collaboration_requests()
        await second.drain_collaboration_requests()
