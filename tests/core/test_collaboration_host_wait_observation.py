"""Native wait work remains attached beyond every nested public read bound."""

import asyncio

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import registration as participant_registration
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import CollaborationHost, HostOwnershipLimits, HostRegistration, HostWaitRule
from cayu.collaboration.requests import RequestControl

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("failure_boundary", ["read", "authorize-first", "authorize-later"])
async def test_failed_native_wait_read_releases_local_turn(
    native_stores, monkeypatch, failure_boundary
):
    store = native_stores[0]
    reg = participant_registration(
        limits=participant_registration().bootstrap.limits.model_copy(update={"events": 512})
    )
    app, resolver, values = await public_setup(store, session_store=native_stores[1], reg=reg)
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    second = await app.accept_collaboration_request(
        values[4].model_copy(update={"operation": values[1].operation("second-request")}),
        context=resolver.context,
    )
    first = wait_for(accepted, values[1], targets=(accepted.expected, second.expected))
    sibling = first.model_copy(update={"operation": values[1].operation("unrelated-wait")})
    for wait in (first, sibling):
        await app.register_collaboration_wait(wait, context=resolver.context)
    pending = (await app.list_collaboration_waits(context=CONTEXT)).items
    rules = tuple(
        HostWaitRule(
            next(item.recovery for item in pending if item.recovery.operation == wait.operation),
            resolver.context,
        )
        for wait in (first, sibling)
    )
    coordinator = app._wait_coordinator
    initial = await coordinator.inspect(first, context=resolver.context)
    observe, load = coordinator._observe_owned, store._load_wait
    primary = ExceptionGroup("native read failed", [OSError("read"), RuntimeError("cleanup")])
    observing = False
    rejected = False
    observed = set()
    authorization_calls = 0
    native_failures = []
    requests = app._request_coordinator
    authorize, held = requests._authorize_retained_source, requests._held

    async def track_observation(wait, **kwargs):
        nonlocal observing
        observing = True
        try:
            result = await observe(wait, **kwargs)
            observed.add(wait.operation)
            return result
        finally:
            observing = False

    async def fail_native_read(initialized, wait, **kwargs):
        nonlocal rejected
        if observing and not rejected and failure_boundary == "read":
            assert wait.operation == first.operation
            rejected = True
            raise primary
        return await load(initialized, wait, **kwargs)

    async def fail_authorization(value, *, mode):
        nonlocal rejected
        wanted = 2 if failure_boundary == "authorize-later" else 1
        if (
            observing
            and not rejected
            and failure_boundary != "read"
            and mode == "authorize_retained"
            and authorization_calls == wanted
        ):
            rejected = True
            raise primary
        return await held(value, mode=mode)

    async def observe_authorization(*args, **kwargs):
        nonlocal authorization_calls
        if observing:
            authorization_calls += 1
        try:
            return await authorize(*args, **kwargs)
        except Exception as error:
            # Keep the native owner's privacy-safe error, not its raw dependency.
            native_failures.append(error)
            raise

    monkeypatch.setattr(coordinator, "_observe_owned", track_observation)
    monkeypatch.setattr(store, "_load_wait", fail_native_read)
    monkeypatch.setattr(requests, "_held", fail_authorization)
    monkeypatch.setattr(requests, "_authorize_retained_source", observe_authorization)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            wait_rules=rules,
            observation_timeout_s=30,
            shutdown_timeout_s=30,
        ),
    )
    try:
        async with asyncio.timeout(180):
            with pytest.raises(Exception) as caught:
                while True:
                    await host.service_once()
                    await asyncio.sleep(0.01)
            assert caught.value is (primary if failure_boundary == "read" else native_failures[0])
            assert authorization_calls == (1 if failure_boundary == "authorize-first" else 2)
            assert rejected and not observed
            assert host._owned.has_slot("maintenance")
            assert host.inspect().failed == host.inspect().uncertain == 0
            assert await coordinator.inspect(first, context=resolver.context) == initial
            # The source responsibility remains pending. Only the failed local
            # turn ended; fresh passes must still use normal observation gates.
            await app.control_collaboration_request(
                RequestControl(
                    operation=values[1].operation("close-after-read-failure"),
                    expected=accepted.expected,
                    expected_revision=1,
                    kind="cancel",
                ),
                context=resolver.context,
            )
            while observed != {first.operation, sibling.operation}:
                await host.service_once()
                await asyncio.sleep(0.01)
            for wait in (first, sibling):
                snapshot = await coordinator.inspect(wait, context=resolver.context)
                assert snapshot.state == "elected" and not snapshot.source_pins
            assert not (await host.aclose()).pending
    finally:
        await host.aclose()


@pytest.mark.parametrize(
    "boundary",
    [
        "authorize_retained",
        "register_observation",
        "read_observation_source",
        "_load_wait",
        "_record_wait_evidence",
        "_release_wait_sources",
    ],
)
async def test_host_wait_keeps_nested_observation_until_ack(native_stores, monkeypatch, boundary):
    store = native_stores[0]
    app, resolver, values = await public_setup(store, session_store=native_stores[1])
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    await app.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("close-host-wait-source"),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    wait = wait_for(accepted, values[1])
    await app.register_collaboration_wait(wait, context=resolver.context)
    recovery = (await app.list_collaboration_waits(context=CONTEXT)).items[0].recovery
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    requests = app._request_coordinator
    previous_timeout = requests._owners.observation_timeout

    async def hold_ack(result):
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return result

    if boundary.startswith("_"):
        operation = getattr(store, boundary)

        async def blocked(*args, **kwargs):
            return await hold_ack(await operation(*args, **kwargs))

        monkeypatch.setattr(store, boundary, blocked)
    else:
        held = requests._held

        async def blocked_source(value, *, mode):
            result = await held(value, mode=mode)
            return await hold_ack(result) if mode == boundary else result

        monkeypatch.setattr(requests, "_held", blocked_source)

    monkeypatch.setattr(requests._owners, "observation_timeout", 0.02)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            wait_rules=(HostWaitRule(recovery, resolver.context),),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01,
        ),
    )
    running = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        running.cancel()
        running.cancel()
        assert running.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelled() and running.cancelling() == 2
        await asyncio.sleep(0.06)
        for _ in range(4):
            observed = await host.service_once()
            assert observed.uncertain == 1
            assert observed.failed == observed.source_failures == 0
        assert calls == [True]
        closed = await host.aclose()
        assert closed.pending and closed.uncertain == 1 and closed.failed == 0
    finally:
        # Restore the client bound before public verification, not before the
        # assertion that retained runtime work outlives that bound.
        requests._owners.observation_timeout = previous_timeout
        release.set()
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        async with asyncio.timeout(60):
            while (await host.aclose()).pending:
                for result in host._owned.inspect().completed:
                    if result.error is not None:
                        raise result.error
                await asyncio.sleep(0.01)
    observed = await app.observe_collaboration_wait(wait, context=resolver.context)
    assert observed.state == "elected" and observed.election.result == "failure"
    assert not observed.source_pins
    assert not requests._owners.pending
