"""Host expiry reconciles an exact committed control, not task termination."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_participant_identity import stores as stores

from cayu._exception_groups import iter_exception_tree
from cayu.collaboration._contracts import ExactConflict, ExactMatch
from cayu.collaboration._host import CollaborationHost, _HostRegistration
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_requests import _RequestMaintenanceSource
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestControl

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("competing_kind", ["cancel", "expire"])
async def test_expiry_losing_to_another_control_releases_local_capacity(
    stores, monkeypatch, competing_kind
):
    application, resolver, values = await public_setup(stores())
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    store, _ = application._participant_coordinator._ready()
    transaction = store._transaction
    clock_ms = accepted.expected.intent.selection.expires_at_ms

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return clock_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", at_deadline)
    other_store = stores()
    if other_store is not store:
        other_transaction = other_store._transaction

        @asynccontextmanager
        async def other_at_deadline(scope, *, write):
            async with other_transaction(scope, write=write) as tx:

                async def now_ms():
                    return clock_ms

                tx.now_ms = now_ms
                yield tx

        monkeypatch.setattr(other_store, "_transaction", other_at_deadline)
    other = app(
        other_store,
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await other.initialize_collaboration()
    coordinator = application._request_coordinator
    control = coordinator.control
    entered, release = asyncio.Event(), asyncio.Event()
    failures = []

    async def pause_host_control(command, **kwargs):
        if command.operation.caller_key.startswith("host-request-expiry:"):
            entered.set()
            await release.wait()
        try:
            return await control(command, **kwargs)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(coordinator, "control", pause_host_control)
    host = CollaborationHost(
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
    try:
        async with asyncio.timeout(30):
            while not entered.is_set():
                await host.service_once()
        winner = await other.control_collaboration_request(
            RequestControl(
                operation=values[1].operation("competing-terminal-control"),
                expected=accepted.expected,
                expected_revision=1,
                kind=competing_kind,
            ),
            context=resolver.context,
        )
        assert winner.state == "expired"
        release.set()
        reported = []
        async with asyncio.timeout(30):
            while not reported:
                try:
                    await host.service_once()
                except Exception as error:
                    reported.append(error)
        assert len(failures) == 1 and reported == failures
        await host.service_once()
        assert host.inspect().active == host.inspect().failed == 0
        assert host.inspect().uncertain == 0
        another = await application.accept_collaboration_request(
            values[4].model_copy(update={"operation": values[1].operation("unrelated-request")}),
            context=resolver.context,
        )
        clock_ms = another.expected.intent.selection.expires_at_ms
        async with asyncio.timeout(30):
            while True:
                await host.service_once()
                next_state = await application.inspect_collaboration_request(
                    another.expected, context=resolver.context
                )
                if next_state.state == "expired":
                    break
        assert (await host.aclose()).pending == 0
        current = await application.inspect_collaboration_request(
            accepted.expected, context=resolver.context
        )
        assert current.terminal == winner
    finally:
        release.set()
        await host.aclose()
        await application.drain_collaboration_requests()


@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_expiry_lost_ack_reconciles_without_hiding_failure(
    stores, monkeypatch, cancel_observer, delayed_readback=False
):
    application, resolver, values = await public_setup(stores())
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    store, _ = application._participant_coordinator._ready()
    transaction = store._transaction
    deadline = accepted.expected.intent.selection.expires_at_ms

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return deadline

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", at_deadline)
    coordinator = application._request_coordinator
    held = coordinator._held
    lost_ack = ConnectionError("expiry acknowledgement lost")
    entered, release = asyncio.Event(), asyncio.Event()
    dispatches = 0
    lookup_attempts = 0

    async def fail_after_commit(value, *, mode):
        nonlocal dispatches, lookup_attempts
        if mode == "lookup_control":
            lookup_attempts += 1
            if delayed_readback and lookup_attempts == 1:
                raise OSError("readback temporarily unavailable")
        if mode == "lookup_control" and cancel_observer:
            entered.set()
            await release.wait()
        result = await held(value, mode=mode)
        if mode == "control":
            dispatches += 1
            raise lost_ack
        return result

    monkeypatch.setattr(coordinator, "_held", fail_after_commit)
    native_failures = []
    native_control = coordinator.control

    async def observe_native_failure(*args, **kwargs):
        try:
            return await native_control(*args, **kwargs)
        except Exception as error:
            # The native boundary deliberately sanitizes private dependencies.
            # The host must preserve the exact graph it receives, not restore
            # the raw backend error across that established privacy boundary.
            native_failures.append(error)
            raise

    monkeypatch.setattr(coordinator, "control", observe_native_failure)
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
    reported = []
    runner = None
    try:
        if cancel_observer:
            runner = asyncio.create_task(host.run())
            await asyncio.wait_for(entered.wait(), 30)
            runner.cancel()
            runner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner
            assert runner.cancelled() and runner.cancelling() == 2
            assert (await host.aclose()).uncertain == 1
            # Source disappearance is not settlement. The exact read is still
            # blocked, and must remain owned even though expiry committed.
            assert not (
                await application.list_due_collaboration_requests(context=resolver.context)
            ).items
            release.set()
            async with asyncio.timeout(30):
                while not any(item.reconciled for item in host._owned.inspect().completed):
                    await asyncio.sleep(0.001)
            # Cancel a later close observer after native completion, precisely
            # at the following discovery-drain await. The original failure
            # must not have been discarded when the local slot was released.
            drain_entered = asyncio.Event()
            original_close = host._reads.close
            hold_drain = True

            async def pause_drain(timeout):
                if hold_drain:
                    drain_entered.set()
                    await asyncio.Event().wait()
                return await original_close(timeout)

            monkeypatch.setattr(host._reads, "close", pause_drain)
            closing = asyncio.create_task(host.aclose())
            await asyncio.wait_for(drain_entered.wait(), 30)
            closing.cancel()
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert closing.cancelled() and closing.cancelling() == 2
            assert host.inspect().uncertain == 1
            assert host._owned.inspect().completed[0].error is native_failures[0]
            hold_drain = False
            async with asyncio.timeout(30):
                while host.inspect().pending:
                    try:
                        await host.aclose()
                    except Exception as error:
                        reported.append(error)
                    await asyncio.sleep(0)
        else:
            async with asyncio.timeout(30):
                while not reported:
                    try:
                        await host.service_once()
                    except Exception as error:
                        reported.append(error)
            # Reobserving the host cannot repeat the failure or dispatch.
            await host.service_once()
        assert len(reported) == 1
        assert len(native_failures) == 1
        assert reported[0] is native_failures[0]
        if delayed_readback:
            # The public exact reader classifies this transient dependency
            # failure as ExactUnavailable, not an escaping exception.
            assert lookup_attempts >= 2
        assert sum(error is native_failures[0] for error in iter_exception_tree(reported[0])) == 1
        assert host.inspect().uncertain == 0 and host.inspect().failed == 0
        retained = await application.inspect_collaboration_request(
            accepted.expected, context=resolver.context
        )
        assert retained.state == "expired" and retained.revision == 2
        control = retained.terminal.expected.intent
        assert isinstance(
            await coordinator.lookup_control_intent(control, context=resolver.context), ExactMatch
        )
        for changes in ({"expected_revision": 2}, {"kind": "cancel"}):
            assert isinstance(
                await coordinator.lookup_control_intent(
                    control.model_copy(update=changes), context=resolver.context
                ),
                ExactConflict,
            )
        assert dispatches == 1
        assert (
            await application.inspect_participant(values[3].reference, context=CONTEXT)
        ).outstanding_obligations == 0
    finally:
        release.set()
        if runner is not None and not runner.done():
            runner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runner
        async with asyncio.timeout(30):
            while True:
                try:
                    pending = (await host.aclose()).pending
                except Exception:
                    pending = host.inspect().pending
                if not pending:
                    break
                await asyncio.sleep(0.01)
        await application.drain_collaboration_requests()


async def test_expiry_recovers_after_initial_readback_failure(stores, monkeypatch):
    await test_expiry_lost_ack_reconciles_without_hiding_failure(
        stores, monkeypatch, cancel_observer=False, delayed_readback=True
    )


async def test_expiry_failure_without_committed_control_stays_fenced(stores, monkeypatch):
    application, resolver, values = await public_setup(stores())
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    store, _ = application._participant_coordinator._ready()
    transaction = store._transaction

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return accepted.expected.intent.selection.expires_at_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", at_deadline)
    coordinator = application._request_coordinator
    held = coordinator._held
    calls = 0

    async def fail_before_commit(value, *, mode):
        nonlocal calls
        if mode == "control":
            calls += 1
            raise ConnectionError("control did not commit")
        return await held(value, mode=mode)

    monkeypatch.setattr(coordinator, "_held", fail_before_commit)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            request_maintenance_sources=(_RequestMaintenanceSource(resolver.context),),
            observation_timeout_s=30,
            shutdown_timeout_s=0.01,
        ),
    )
    try:
        async with asyncio.timeout(30):
            while not host.inspect().failed:
                await host.service_once()
        first = host._owned.inspect().completed[0]
        assert first.error is not None and not first.reconciled
        assert first.value is None
        await host.service_once()
        assert calls == 1
        closed = await host.aclose()
        assert closed.failed == 1 and closed.uncertain == 1
        retained = await application.inspect_collaboration_request(
            accepted.expected, context=resolver.context
        )
        assert retained.state == "open" and retained.terminal is None and retained.revision == 1
        assert host._owned.inspect().completed[0].error is first.error
    finally:
        await host.aclose()
        await application.drain_collaboration_requests()
