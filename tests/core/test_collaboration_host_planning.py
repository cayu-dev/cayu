"""Host planning uses native retained intent without inventing launch authority."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import app as make_app
from tests.core.test_participant_identity import stores as stores
from tests.core.test_request_planning_public import scenario

from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._host import CollaborationHost, _HostRegistration, _PlanningRule
from cayu.collaboration._host_ownership import HostOwnershipLimits

pytestmark = pytest.mark.anyio


async def drive_host_plan(application, command, context):
    """Exercise the same host entrance for inert decisions and native FRESH."""
    registration = _HostRegistration(
        limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
        producer_sources=(),
        producer_rules=(),
        planning_rules=(_PlanningRule(command, context),),
        observation_timeout_s=60,
        shutdown_timeout_s=60,
    )
    async with CollaborationHost(application, registration) as host, asyncio.timeout(180):
        while True:
            state = await host.service_once()
            for outcome in host._owned.inspect().completed:
                if outcome.error is not None:
                    raise outcome.error
            assert state.source_failures == 0, host._source_errors
            found = await application.lookup_collaboration_plan(command, context=context)
            if (
                isinstance(found, ExactMatch)
                and found.receipt.state in ("admitted", "declined", "deferred")
                and not found.receipt.pending_stages
            ):
                return found.receipt


@pytest.mark.parametrize("competing", [False, True])
async def test_host_retains_exact_plan_without_model_dispatch(stores, monkeypatch, competing):
    application, resolver, command, _, provider = await scenario(stores(), "decline")
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    context = resolver.recipient.context
    assert isinstance(
        await application.lookup_collaboration_plan(command, context=context), ExactNotFound
    )
    if competing:
        other = make_app(
            stores(),
            application._participant_coordinator._registration,
            collaboration_requests=application._request_coordinator._registration,
        )
        await other.initialize_collaboration()
        monkeypatch.setattr(other._request_coordinator._owners, "observation_timeout", 60)
        retained, second = await asyncio.gather(
            drive_host_plan(application, command, context),
            drive_host_plan(other, command, context),
        )
        assert second == retained
    else:
        retained = await drive_host_plan(application, command, context)
    assert retained.state == "declined"
    assert not provider.requests
    # A new host observes the same terminal decision without evaluating policy
    # again or inventing a successor generation.
    assert await drive_host_plan(application, command, context) == retained
    assert not provider.requests


async def test_planning_observer_cancel_and_close_retain_committed_work(stores, monkeypatch):
    from cayu.collaboration import _host

    store = stores()
    application, resolver, command, _, provider = await scenario(store, "decline")
    owners = application._request_coordinator._owners
    monkeypatch.setattr(owners, "observation_timeout", 1.0)
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction
    blocked = False
    reads = []
    lookup = _host.lookup_host_plan

    async def observed_lookup(*args, **kwargs):
        reads.append(True)
        return await lookup(*args, **kwargs)

    monkeypatch.setattr(_host, "lookup_host_plan", observed_lookup)

    @asynccontextmanager
    async def after_commit(scope, *, write):
        nonlocal blocked
        async with original(scope, write=write) as tx:
            yield tx
        if write and not blocked:
            blocked = True
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_transaction", after_commit)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
            producer_sources=(),
            producer_rules=(),
            planning_rules=(_PlanningRule(command, resolver.recipient.context),),
            observation_timeout_s=60,
            shutdown_timeout_s=0.01,
        ),
    )
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        observer.cancel()
        assert observer.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()
        occupied_reads = len(reads)
        for _ in range(8):
            occupied = await host.service_once()
            assert occupied.uncertain == 1 and occupied.failed == 0
        assert len(reads) == occupied_reads
        # Let the ordinary native client observation bound expire while the
        # committed mutation still owns its acknowledgement. The host must keep
        # awaiting that operation, not convert the client timeout into failure.
        await asyncio.sleep(1.25)
        state = await host.aclose()
        assert state.uncertain == 1 and state.failed == 0
        assert not provider.requests
    finally:
        release.set()
        async with asyncio.timeout(20):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
        if not observer.done():
            await observer
    monkeypatch.setattr(owners, "observation_timeout", 60)
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch) and found.receipt.state == "declined"
    assert not provider.requests


async def test_planning_lost_ack_drains_after_exact_readback_recovers(stores, monkeypatch):
    from cayu.collaboration import _host_planning
    from cayu.collaboration._contracts import ExactUnavailable

    application, resolver, command, _, provider = await scenario(stores(), "decline")
    context = resolver.recipient.context
    primary = ExceptionGroup("planning acknowledgement lost", [RuntimeError("lost ack")])
    unavailable = False
    dispatches = []
    service = _host_planning.service_application_plan
    lookup = _host_planning.lookup_host_plan

    async def committed_then_failed(*args, **kwargs):
        nonlocal unavailable
        record = await service(*args, **kwargs)
        dispatches.append(record)
        unavailable = True
        raise primary

    async def temporarily_unavailable(*args, **kwargs):
        if unavailable:
            return ExactUnavailable()
        return await lookup(*args, **kwargs)

    monkeypatch.setattr(_host_planning, "service_application_plan", committed_then_failed)
    monkeypatch.setattr(_host_planning, "lookup_host_plan", temporarily_unavailable)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
            producer_sources=(),
            producer_rules=(),
            planning_rules=(_PlanningRule(command, context),),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01,
        ),
    )
    failures = []
    try:
        async with asyncio.timeout(120):
            while not host.inspect().failed:
                await host.service_once()
        assert len(dispatches) == 1
        assert (await host.aclose()).pending
        assert host.inspect().failed == 1
        # A separate public read proves the native decision committed. The
        # original host still cannot discharge its slot from unavailable evidence.
        found = await application.lookup_collaboration_plan(command, context=context)
        assert isinstance(found, ExactMatch) and found.receipt.state == "declined"
    finally:
        unavailable = False
        async with asyncio.timeout(120):
            while True:
                try:
                    if not (await host.aclose()).pending:
                        break
                except ExceptionGroup as error:
                    failures.append(error)
                await asyncio.sleep(0.01)
    assert failures == [primary]
    assert len(dispatches) == 1
    assert not provider.requests
    assert host.inspect().failed == 0
