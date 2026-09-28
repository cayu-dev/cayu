"""Dependent native inspections retain ownership without monopolizing host passes."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerDisclosure,
    _ProducerExecutionRule,
    _ProducerOutputRule,
    _ProducerSource,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_execution import HostProducerExecution
from cayu.collaboration._host_requests import _RequestMaintenanceSource
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.providers.base import ModelStreamEvent

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("withdraw_authority", [False, True])
async def test_slow_preparation_resumes_in_next_explicit_pass_without_reconstruction(
    native_stores, monkeypatch, withdraw_authority
):
    from cayu.collaboration import _host_producer_execution

    app, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores,
        with_exports=True,
        planned=True,
        request_ttl_ms=900_000,
        provider_events=(
            (ModelStreamEvent.text_delta("Prepared once."), ModelStreamEvent.completed()),
        ),
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    without_execution = resolver.recipient.resolution
    authorize_execution(resolver)
    page = await app.pending_producer_outputs(command.admission.prepared.recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    recover = _host_producer_execution.recover_registered_execution
    entered, release = asyncio.Event(), asyncio.Event()
    reconstructions = 0

    async def slow_preparation(*args, **kwargs):
        nonlocal reconstructions
        result = await recover(*args, **kwargs)
        reconstructions += 1
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(_host_producer_execution, "recover_registered_execution", slow_preparation)
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(command.admission.prepared.recipient, CONTEXT),),
            producer_rules=(),
            producer_execution_rules=(
                _ProducerExecutionRule(
                    HostProducerExecution(recovery=token), CONTEXT, resolver.recipient.context
                ),
            ),
            observation_timeout_s=0.02,
            shutdown_timeout_s=0.01,
        ),
    )
    running = asyncio.create_task(host.run())
    try:
        async with asyncio.timeout(120):
            while not entered.is_set():
                if running.done():
                    await running
                if host._source_errors:
                    raise ExceptionGroup("Host source failed", list(host._source_errors.values()))
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelled()
        await asyncio.sleep(0.04)
        release.set()
        await asyncio.sleep(0.04)
        assert reconstructions == 1
        assert not provider.requests
        assert host.inspect().uncertain == 1
        if withdraw_authority:
            resolver.recipient.resolution = without_execution
        rejected = None
        async with asyncio.timeout(120):
            while not host.inspect().serviced:
                await host.service_once()
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        if not withdraw_authority:
                            raise outcome.error
                        rejected = outcome.error
                if rejected is not None:
                    break
                await asyncio.sleep(0.01)
        assert reconstructions == 1
        if withdraw_authority:
            assert isinstance(rejected, PermissionError)
            assert not provider.requests
            assert host.inspect().uncertain == host.inspect().failed == 1
        else:
            assert rejected is None
            assert len(provider.requests) == 1
    finally:
        release.set()
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        async with asyncio.timeout(60):
            while (closed := await host.aclose()).pending:
                if closed.failed:
                    # A retained failure is not running work; do not mask the
                    # primary test failure with a second drain timeout.
                    break
                await asyncio.sleep(0.01)


async def test_slow_native_inspection_retains_one_read_and_services_other_families(
    native_stores, monkeypatch
):
    app, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores, with_exports=True, request_ttl_ms=900_000
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    page = await app.pending_producer_outputs(command.admission.prepared.recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    coordinator = app._request_coordinator
    dependency, due = coordinator._dependency, coordinator.due
    entered, release = asyncio.Event(), asyncio.Event()
    inspections = other_reads = 0

    async def delayed_ack(operation):
        nonlocal inspections
        result = await dependency(operation)
        if operation.__qualname__ == "_lookup_producer.<locals>.lookup":
            inspections += 1
            entered.set()
            await release.wait()
        return result

    async def count_due(**kwargs):
        nonlocal other_reads
        result = await due(**kwargs)
        if entered.is_set():
            other_reads += 1
        return result

    monkeypatch.setattr(coordinator, "_dependency", delayed_ack)
    monkeypatch.setattr(coordinator, "due", count_due)
    monkeypatch.setattr(coordinator._owners, "observation_timeout", 0.01)
    access = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(command.admission.prepared.recipient, CONTEXT),),
            producer_rules=(),
            producer_output_rules=(
                _ProducerOutputRule(
                    token,
                    CONTEXT,
                    access,
                    tuple(
                        _ProducerDisclosure(item.operation, access) for item in command.destinations
                    ),
                ),
            ),
            request_maintenance_sources=(_RequestMaintenanceSource(resolver.sender.context),),
            observation_timeout_s=0.02,
            shutdown_timeout_s=0.01,
        ),
    )
    running = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 60)
        running.cancel()
        running.cancel()
        assert running.cancelling() == 2
        with pytest.raises(asyncio.CancelledError):
            await running
        assert running.cancelled()
        async with asyncio.timeout(30):
            while other_reads < 2:
                await host.service_once()
                await asyncio.sleep(0.01)
        assert inspections == 1
        assert not release.is_set()
        assert not provider.requests
        assert host.inspect().source_failures == host.inspect().failed == 0
        closed = await host.aclose()
        assert closed.discovery_pending >= 1
        assert not coordinator._owners.closed
    finally:
        release.set()
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        async with asyncio.timeout(60):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
    assert inspections == 1
    assert not coordinator._owners.pending
    assert not provider.requests
