"""Native decisions outlive lost acknowledgements without leaking host slots."""

import asyncio

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu import CollaborationHost, HostOwnershipLimits, HostRegistration, HostWaitRule
from cayu.collaboration._host import _ProducerExecutionRule, _ProducerSource
from cayu.collaboration._host_producer_execution import HostProducerExecution
from cayu.collaboration.requests import RequestControl
from cayu.providers.base import ModelStreamEvent

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("action", ["publish_answer", "publish_failure"])
async def test_competing_request_close_excludes_host_publication_turn(
    native_stores, monkeypatch, action
):
    from cayu import HostProducerMaintenance, HostProducerMaintenanceRule, HostProducerSource
    from cayu.collaboration import _host_producer_maintenance as maintenance
    from cayu.collaboration._contracts import ExactMatch
    from cayu.collaboration._producer_registration import register_producer_output
    from cayu.collaboration.exports import SessionExportAccessContext

    app, resolver, admission, provider, _, initialized, command, execution = await output_scenario(
        native_stores, with_exports=True
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await register_producer_output(app, command, execution, context=resolver.recipient.context)
    page = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    publish = maintenance.publish_producer_outcome
    entered, release = asyncio.Event(), asyncio.Event()
    failures = []

    async def paused_publish(*args, **kwargs):
        entered.set()
        await release.wait()
        try:
            return await publish(*args, **kwargs)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(maintenance, "publish_producer_outcome", paused_publish)
    intent = HostProducerMaintenance(
        recovery=token,
        action=action,
        destination=command.destinations[0].operation if action == "publish_answer" else None,
    )
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(HostProducerSource(admission.prepared.recipient, CONTEXT),),
            producer_rules=(
                HostProducerMaintenanceRule(
                    intent,
                    CONTEXT,
                    SessionExportAccessContext(
                        principal=resolver.recipient.context.principal,
                        mandate=resolver.recipient.context,
                    ),
                ),
            ),
            observation_timeout_s=60,
            shutdown_timeout_s=60,
        ),
    )
    try:
        async with asyncio.timeout(120):
            while not entered.is_set():
                await host.service_once()
        current = await app.inspect_collaboration_request(
            admission.expected, context=resolver.sender.context
        )
        await app.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("competing-close"),
                expected=admission.expected,
                expected_revision=current.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
        after_native_control = await app.inspect_producer_output(command, context=CONTEXT)
        assert isinstance(after_native_control, ExactMatch)
        release.set()
        reported = []
        async with asyncio.timeout(120):
            while not reported:
                try:
                    await host.service_once()
                except Exception as error:
                    reported.append(error)
        assert host._owned.has_slot("maintenance")
        async with asyncio.timeout(120):
            while True:
                try:
                    if not (await host.aclose()).pending:
                        break
                except Exception as error:
                    reported.append(error)
                await asyncio.sleep(0.01)
        assert len(failures) == 1 and reported == failures
        assert not provider.requests
        assert host.inspect().uncertain == host.inspect().failed == 0
        remaining = await app.inspect_producer_output(command, context=CONTEXT)
        assert isinstance(remaining, ExactMatch)
        assert remaining.receipt.request_state == "cancelled"
        # Native cancellation can itself settle inert production. Host
        # reconciliation must neither invent nor change that cleanup evidence.
        assert remaining == after_native_control
    finally:
        release.set()
        await host.aclose()


async def recover_same_host(host, primary, entered, release):
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 120)
        assert host.inspect().failed == host.inspect().uncertain == 1
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        # Cancellation of the foreground observer neither retries the effect
        # nor releases its slot while the exact read is still in flight.
        for _ in range(2):
            await host.service_once()
            assert host.inspect().uncertain == 1
        release.set()
        async with asyncio.timeout(120):
            with pytest.raises(Exception) as caught:
                while True:
                    await host.service_once()
                    await asyncio.sleep(0.001)
        assert caught.value is primary
        assert host.inspect().failed == host.inspect().uncertain == 0
        assert host._owned.has_slot("execution") and host._owned.has_slot("maintenance")
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        async with asyncio.timeout(120):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.001)


@pytest.mark.parametrize("preflight", [False, True])
async def test_wait_commit_ack_loss_recovers_on_same_host(native_stores, monkeypatch, preflight):
    from tests.core.test_participant_identity import registration

    # Two native waits each reserve their terminal-event capacity up front.
    reg = registration(limits=registration().bootstrap.limits.model_copy(update={"events": 512}))
    app, resolver, values = await public_setup(
        native_stores[0], session_store=native_stores[1], reg=reg
    )
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    await app.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("close-host-reconciliation-source"),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    wait = wait_for(accepted, values[1])
    await app.register_collaboration_wait(wait, context=resolver.context)
    recovery = (await app.list_collaboration_waits(context=CONTEXT)).items[0].recovery
    rules = [HostWaitRule(recovery, resolver.context)]
    observed_waits = set()
    if preflight:
        sibling = wait.model_copy(update={"operation": values[1].operation("independent-wait")})
        await app.register_collaboration_wait(sibling, context=resolver.context)
        sibling_recovery = next(
            item.recovery
            for item in (await app.list_collaboration_waits(context=CONTEXT)).items
            if item.recovery.operation == sibling.operation
        )
        rules.append(HostWaitRule(sibling_recovery, resolver.context))
    coordinator = app._wait_coordinator
    observe, inspect = coordinator._observe_owned, coordinator.inspect
    primary = ConnectionError("wait acknowledgement lost")
    entered, release = asyncio.Event(), asyncio.Event()
    mutations = reads = 0

    async def commit_then_raise(*args, **kwargs):
        nonlocal mutations
        await observe(*args, **kwargs)
        mutations += 1
        raise primary

    async def unavailable_then_blocked(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return None
        result = await inspect(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    if preflight:
        from cayu.collaboration import _host_waits as adapter

        lookup = adapter.resolve_wait

        async def fail_preflight(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise primary
            return await lookup(*args, **kwargs)

        monkeypatch.setattr(adapter, "resolve_wait", fail_preflight)

        async def track_wait(candidate, **kwargs):
            result = await observe(candidate, **kwargs)
            observed_waits.add(candidate.operation)
            return result

        monkeypatch.setattr(coordinator, "_observe_owned", track_wait)
    else:
        monkeypatch.setattr(coordinator, "_observe_owned", commit_then_raise)
        monkeypatch.setattr(coordinator, "inspect", unavailable_then_blocked)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            wait_rules=tuple(rules),
            observation_timeout_s=0.1,
        ),
    )
    if preflight:
        await retry_preflight(host, primary, completed=lambda: len(observed_waits) == 2)
        assert mutations == 0 and reads >= 2
    else:
        await recover_same_host(host, primary, entered, release)
        assert mutations == 1 and reads == 2
    retained = await inspect(wait, context=resolver.context)
    assert retained.state == "elected" and not retained.source_pins


@pytest.mark.parametrize("preflight", [False, True, "cancel-owned"])
async def test_producer_release_ack_loss_recovers_without_redispatch(
    native_stores, monkeypatch, preflight
):
    from cayu.collaboration import _host_producer_execution as adapter

    app, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores,
        with_exports=True,
        planned=True,
        request_ttl_ms=900_000,
        provider_events=(
            (ModelStreamEvent.text_delta("One invocation."), ModelStreamEvent.completed()),
        ),
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    authorize_execution(resolver)
    participant = command.admission.prepared.recipient
    page = await app.pending_producer_outputs(participant, context=CONTEXT)
    recovery = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    original = adapter._release_proof
    primary = ConnectionError("release acknowledgement lost")
    entered, release = asyncio.Event(), asyncio.Event()
    dispatched = asyncio.Event()
    native_tasks = []
    reads = 0

    async def lost_then_blocked(*args, **kwargs):
        nonlocal reads
        result = await original(*args, **kwargs)
        reads += 1
        if reads == 1:
            if preflight == "cancel-owned":
                native_tasks.append(asyncio.current_task())
                dispatched.set()
                await asyncio.Event().wait()
            raise primary
        entered.set()
        await release.wait()
        return result

    if preflight is True:
        lookup = adapter.inspect_producer_output

        async def fail_preflight(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 1:
                raise primary
            return await lookup(*args, **kwargs)

        monkeypatch.setattr(adapter, "inspect_producer_output", fail_preflight)
    else:
        monkeypatch.setattr(adapter, "_release_proof", lost_then_blocked)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(participant, CONTEXT),),
            producer_rules=(),
            producer_execution_rules=(
                _ProducerExecutionRule(
                    HostProducerExecution(recovery=recovery),
                    CONTEXT,
                    resolver.recipient.context,
                ),
            ),
            observation_timeout_s=0.1,
        ),
    )
    if preflight == "cancel-owned":
        observer = asyncio.create_task(host.run())
        try:
            await asyncio.wait_for(dispatched.wait(), 120)
            native_tasks[0].cancel()
            native_tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(observer, 120)
            assert observer.cancelled()
            assert native_tasks[0].cancelling() == 2
            assert native_tasks[0].cancelled()
            assert host.inspect().uncertain == 1
            assert not host._owned.has_slot("execution")
            for _ in range(2):
                await host.service_once()
                assert host.inspect().uncertain == 1
            await asyncio.wait_for(entered.wait(), 120)
            release.set()
            async with asyncio.timeout(120):
                while host.inspect().uncertain:
                    await host.service_once()
            assert host._owned.has_slot("execution")
            # Historical cancellation cannot recur on later service or drain.
            await host.service_once()
        finally:
            release.set()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await host.aclose()
    elif preflight:
        await retry_preflight(host, primary)
    else:
        await recover_same_host(host, primary, entered, release)
    assert reads == 2 and len(provider.requests) == 1


async def retry_preflight(host, primary, *, completed=None):
    async with host, asyncio.timeout(180):
        with pytest.raises(Exception) as caught:
            await host.run()
        assert caught.value is primary
        assert host.inspect().uncertain == host.inspect().failed == 0
        assert host._owned.has_slot("execution") and host._owned.has_slot("maintenance")
        while True:
            state = await host.service_once()
            assert host.inspect().failed == 0
            if state.serviced and (completed is None or completed()):
                break
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_delivery_maintenance_recovers_late_readback(backend, tmp_path, request, monkeypatch):
    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    from cayu.collaboration._host_clarification_maintenance import _ClarificationMaintenanceSource

    exercised = []

    async def install_delivery_probe(app, *, context):
        coordinator = app._clarification_coordinator
        # Capture a genuine pending page before delivery; hold that discovery
        # acknowledgement until the native receiving decision has committed.
        page = await app.list_pending_clarification_deliveries(context=context)
        assert len(page.items) == 1
        deliver = coordinator.deliver

        async def after_delivery(*args, **kwargs):
            result = await deliver(*args, **kwargs)
            if exercised or result.status == "pending":
                return result
            exercised.append(result)
            reconcile = coordinator.reconcile_delivery
            inspect = coordinator._inspect_maintenance_owned
            primary = ConnectionError("delivery maintenance acknowledgement lost")
            entered, release = asyncio.Event(), asyncio.Event()
            calls = reads = 0

            async def stale_scan(**query):
                return page

            async def commit_then_raise(*args, **kwargs):
                nonlocal calls
                receipt = await reconcile(*args, **kwargs)
                assert receipt == result
                calls += 1
                raise primary

            async def unavailable_then_blocked(*args, **kwargs):
                nonlocal reads
                reads += 1
                if reads == 1:
                    return None
                receipt = await inspect(*args, **kwargs)
                assert receipt == result
                entered.set()
                await release.wait()
                return receipt

            with monkeypatch.context() as patch:
                patch.setattr(coordinator, "pending_deliveries", stale_scan)
                patch.setattr(coordinator, "reconcile_delivery", commit_then_raise)
                patch.setattr(coordinator, "_inspect_maintenance_owned", unavailable_then_blocked)
                host = CollaborationHost(
                    app,
                    HostRegistration(
                        limits=HostOwnershipLimits(1, 1, 2, 262144),
                        producer_sources=(),
                        producer_rules=(),
                        clarification_maintenance_sources=(
                            _ClarificationMaintenanceSource("deliveries", context),
                        ),
                        observation_timeout_s=0.1,
                    ),
                )
                await recover_same_host(host, primary, entered, release)
            assert calls == 1 and reads == 2
            return result

        monkeypatch.setattr(coordinator, "deliver", after_delivery)

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        maintenance_driver=install_delivery_probe,
        journey_ttl_ms=900_000,
    )
    assert len(exercised) == 1
