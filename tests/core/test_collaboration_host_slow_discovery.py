"""A delayed native source reply cannot block unrelated native maintenance."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import stores as stores
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._host import CollaborationHost, _HostRegistration, _ProducerSource
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_requests import _RequestMaintenanceSource

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("blocked_family", ["request", "clarification"])
async def test_blocked_maintenance_source_preserves_unrelated_expiry(
    native_stores, monkeypatch, blocked_family
):
    from cayu.collaboration._host_clarification_maintenance import (
        _ClarificationMaintenanceSource,
    )

    store = native_stores[0]
    app, resolver, values = await public_setup(store, session_store=native_stores[1])
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    transaction = store._transaction

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return accepted.expected.intent.selection.expires_at_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", at_deadline)
    entered, release = asyncio.Event(), asyncio.Event()
    due = app._request_coordinator.due
    questions = app._clarification_coordinator.due_questions
    deliveries = app._clarification_coordinator.pending_deliveries
    alternate_mandate = resolver.context.mandate.model_copy(update={"object_id": "other-mandate"})
    alternative = resolver.context.model_copy(update={"mandate": alternate_mandate})
    acquire = resolver.acquire

    @asynccontextmanager
    async def authorized_context(context):
        # The trusted fixture issues a second complete root mandate, not a
        # context alias that bypasses the native exact-authority checks.
        async with acquire(resolver.context if context == alternative else context) as resolution:
            if context == alternative:
                leaf = resolution.chain.entries[0].model_copy(
                    update={"reference": alternate_mandate, "root": alternate_mandate}
                )
                resolution = resolution.model_copy(
                    update={"chain": resolution.chain.model_copy(update={"entries": (leaf,)})}
                )
            yield resolution

    monkeypatch.setattr(resolver, "acquire", authorized_context)
    blocked_reads = 0
    healthy_question_reads = 0

    async def blocked(result):
        nonlocal blocked_reads
        blocked_reads += 1
        entered.set()
        await release.wait()
        return result

    async def request_read(**kwargs):
        if blocked_family == "request" and kwargs["context"] == alternative:
            return await blocked(await due(**kwargs))
        await entered.wait()
        return await due(**kwargs)

    async def question_read(**kwargs):
        return await blocked(await questions(**kwargs))

    async def delivery_read(**kwargs):
        nonlocal healthy_question_reads
        await entered.wait()
        result = await deliveries(**kwargs)
        healthy_question_reads += 1
        return result

    monkeypatch.setattr(app._request_coordinator, "due", request_read)
    monkeypatch.setattr(app._clarification_coordinator, "due_questions", question_read)
    monkeypatch.setattr(app._clarification_coordinator, "pending_deliveries", delivery_read)
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            request_maintenance_sources=(
                ((_RequestMaintenanceSource(alternative),) if blocked_family == "request" else ())
                + (_RequestMaintenanceSource(resolver.context),)
            ),
            clarification_maintenance_sources=(
                (
                    _ClarificationMaintenanceSource("questions", CONTEXT),
                    _ClarificationMaintenanceSource("deliveries", CONTEXT),
                )
                if blocked_family == "clarification"
                else ()
            ),
            observation_timeout_s=1,
            shutdown_timeout_s=0.01,
        ),
    )
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        async with asyncio.timeout(60):
            while True:
                observed = await host.service_once()
                assert not observed.source_failures
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                current = await app.inspect_collaboration_request(
                    accepted.expected, context=resolver.context
                )
                if current.state == "expired" and (
                    blocked_family == "request" or healthy_question_reads
                ):
                    break
        assert not release.is_set() and blocked_reads == 1
        assert not app._provider_registry.registrations
        assert (await host.aclose()).discovery_pending >= 1
        assert blocked_reads == 1
    finally:
        release.set()
        entered.set()
        if not observer.done():
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        async with asyncio.timeout(60):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
        await app.drain_collaboration_requests()


@pytest.mark.parametrize("saturation", ["slots", "bytes"])
async def test_saturated_discovery_preserves_producer_cleanup(
    native_stores, monkeypatch, saturation
):
    from tests.core.test_collaboration_host_execution_exclusion import close_request
    from tests.core.test_producer_budget_refusal import authorize_execution
    from tests.core.test_producer_output_contracts import output_scenario

    from cayu.collaboration import _host_discovery
    from cayu.collaboration._contracts import ExactMatch
    from cayu.collaboration._host import _ProducerExecutionRule, _ProducerMaintenanceRule
    from cayu.collaboration._host_producer_execution import HostProducerExecution
    from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
    from cayu.providers.base import ModelStreamEvent

    app, resolver, admission, provider, _, initialized, command, execution = await output_scenario(
        native_stores,
        planned=True,
        with_exports=True,
        request_ttl_ms=900_000,
        provider_events=((ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()),),
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    authorize_execution(resolver)
    async for _ in app.execute_producer_output(
        command, execution, context=CONTEXT, producer_context=resolver.recipient.context
    ):
        pass
    await app.retain_producer_completion(command, context=CONTEXT)
    await close_request(app, initialized, admission.expected, resolver.sender.context, "close-done")
    recipient = admission.prepared.recipient
    page = await app.pending_producer_outputs(recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    unrelated = admission.expected.intent.request.sender
    assert unrelated != recipient
    entered, release = asyncio.Event(), asyncio.Event()
    discover = _host_discovery.pending_producer_outputs

    async def blocked_source(app, participant, **kwargs):
        result = await discover(app, participant, **kwargs)
        if participant == unrelated:
            entered.set()
            await release.wait()
        else:
            await entered.wait()
        return result

    monkeypatch.setattr(_host_discovery, "pending_producer_outputs", blocked_source)
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(
                _ProducerSource(unrelated, CONTEXT),
                _ProducerSource(recipient, CONTEXT),
            ),
            producer_rules=(
                _ProducerMaintenanceRule(
                    HostProducerMaintenance(recovery=token, action="settle"), CONTEXT, None
                ),
            ),
            producer_execution_rules=(
                _ProducerExecutionRule(
                    HostProducerExecution(recovery=token), CONTEXT, resolver.recipient.context
                ),
            ),
            discovery_slots=1 if saturation == "slots" else 4,
            discovery_bytes=131072 if saturation == "bytes" else 512 * 1024,
            observation_timeout_s=0.05,
            shutdown_timeout_s=0.01,
        ),
    )
    try:
        async with asyncio.timeout(120):
            while True:
                state = await host.service_once()
                assert not host._source_errors, host._source_errors
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                if state.serviced:
                    break
        assert entered.is_set() and not release.is_set()
        assert host._reads.pending == 1
        assert not host._owned.pending
        retained = await app.inspect_producer_output(token, context=CONTEXT)
        assert isinstance(retained, ExactMatch)
        assert retained.receipt.cleanup is not None and retained.receipt.cleanup_ack is not None
        assert not (await app.pending_producer_outputs(recipient, context=CONTEXT)).items
        assert len(provider.requests) == 1
        assert (await host.aclose()).discovery_pending >= 1
    finally:
        release.set()
        async with asyncio.timeout(60):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
        await app.drain_collaboration_requests()


async def test_native_discovery_timeout_never_starts_another_read(stores, monkeypatch):
    application, _, values = await public_setup(stores())
    coordinator = application._request_coordinator
    dependency = coordinator._dependency
    entered, release = asyncio.Event(), asyncio.Event()
    reads = 0

    async def blocked_native_reply(operation):
        nonlocal reads
        result = await dependency(operation)
        if operation.__qualname__ == "pending_producer_outputs.<locals>.discover":
            reads += 1
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(coordinator, "_dependency", blocked_native_reply)
    monkeypatch.setattr(coordinator._owners, "observation_timeout", 0.01)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(values[3].reference, CONTEXT),),
            producer_rules=(),
            observation_timeout_s=0.02,
            shutdown_timeout_s=0.01,
        ),
    )
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        observer.cancel()
        observer.cancel()
        assert observer.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()
        await asyncio.sleep(0.03)
        for _ in range(8):
            state = await host.service_once()
            assert state.failed == state.source_failures == 0
        assert reads == 1
        assert len(coordinator._owners.pending) == 1
        assert host._reads.pending == 1
        closed = await host.aclose()
        assert closed.discovery_pending == 1
        assert not coordinator._owners.closed
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        async with asyncio.timeout(30):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
    assert reads == 1
    assert not coordinator._owners.pending
    assert not coordinator._owners.closed


@pytest.mark.parametrize("saturation", ["spare", "slots", "bytes"])
async def test_delayed_producer_discovery_does_not_block_expiry(
    native_stores, monkeypatch, saturation
):
    store = native_stores[0]
    application, resolver, values = await public_setup(store, session_store=native_stores[1])
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    transaction = store._transaction

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with transaction(scope, write=write) as tx:

            async def now_ms():
                return accepted.expected.intent.selection.expires_at_ms

            tx.now_ms = now_ms
            yield tx

    monkeypatch.setattr(store, "_transaction", at_deadline)
    entered, release = asyncio.Event(), asyncio.Event()
    from cayu.collaboration import _host_discovery

    discover_producers = _host_discovery.pending_producer_outputs
    discover_requests = application._request_coordinator.due
    producer_reads = 0

    async def delayed(*args, **kwargs):
        nonlocal producer_reads
        producer_reads += 1
        result = await discover_producers(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    async def after_other_read(**kwargs):
        await entered.wait()
        return await discover_requests(**kwargs)

    monkeypatch.setattr(_host_discovery, "pending_producer_outputs", delayed)
    monkeypatch.setattr(application._request_coordinator, "due", after_other_read)
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(values[3].reference, CONTEXT),),
            producer_rules=(),
            request_maintenance_sources=(_RequestMaintenanceSource(resolver.context),),
            observation_timeout_s=0.1,
            shutdown_timeout_s=0.01,
            discovery_slots=1 if saturation == "slots" else 4,
            discovery_bytes=131072 if saturation == "bytes" else 512 * 1024,
        ),
    )
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 30)
        observer.cancel()
        observer.cancel()
        assert observer.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()
        async with asyncio.timeout(30):
            while True:
                await host.service_once()
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                current = await application.inspect_collaboration_request(
                    accepted.expected, context=resolver.context
                )
                if current.state == "expired":
                    break
        assert not release.is_set()
        assert producer_reads == 1
        assert host._reads.pending == 1
        # Bytes-only saturation leaves slots available; slots-only saturation
        # leaves bytes available. Both must preserve mandatory discovery.
        if saturation == "slots":
            assert host._reads.pending == host._reads._slots
        if saturation == "bytes":
            assert host._reads.pending < host._reads._slots
            assert sum(item[1] for item in host._reads._tasks.values()) == 131072
        assert host.inspect().discovery_pending >= 1
        closed = await host.aclose()
        assert closed.discovery_pending >= 1
        assert producer_reads == 1
    finally:
        release.set()
        async with asyncio.timeout(30):
            while True:
                closed = await host.aclose()
                if not (closed.servicing_pending or closed.discovery_pending or closed.uncertain):
                    break
                await asyncio.sleep(0.01)
        if not observer.done():
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        await application.drain_collaboration_requests()
    assert producer_reads == 1
