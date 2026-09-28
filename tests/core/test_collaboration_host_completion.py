"""Retain a real producer completion beyond cancelled and timed-out observers."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerDisclosure,
    _ProducerMaintenanceRule,
    _ProducerOutputRule,
    _ProducerSource,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._producer_store import completion_operation
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.providers.base import ModelStreamEvent

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("automatic", [False, True])
async def test_host_completion_keeps_native_ownership_after_commit(
    native_stores, monkeypatch, automatic
):
    application, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores,
        with_exports=True,
        request_ttl_ms=900_000,
        provider_events=(
            (ModelStreamEvent.text_delta("Completed."), ModelStreamEvent.completed()),
        ),
    )
    owners = application._request_coordinator._owners
    monkeypatch.setattr(owners, "observation_timeout", 60)
    await application.register_producer_output(
        command, execution, context=resolver.recipient.context
    )
    authorize_execution(resolver)
    async for _ in application.execute_producer_output(
        command, execution, context=CONTEXT, producer_context=resolver.recipient.context
    ):
        pass
    assert len(provider.requests) == 1
    participant = command.admission.prepared.recipient
    page = await application.pending_producer_outputs(participant, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    store = native_stores[0]
    transaction = store._transaction
    key = operation_key(completion_operation(command, application._secret_redactor))
    entered, release = asyncio.Event(), asyncio.Event()
    blocked = False

    @asynccontextmanager
    async def after_completion_commit(scope, *, write):
        nonlocal blocked
        published = False
        async with transaction(scope, write=write) as tx:
            yield tx
            if write and not blocked:
                published = await tx.get("operations", key) is not None
        if published:
            blocked = True
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_transaction", after_completion_commit)
    from cayu.collaboration import _producer_public_control

    retain = _producer_public_control.retain_producer_completion

    async def bounded_public_observation(*args, **kwargs):
        # Only the completion phase uses a short ordinary client observation
        # bound. The native mutation still owns its post-commit acknowledgement.
        previous = owners.observation_timeout
        owners.observation_timeout = 0.05
        try:
            return await retain(*args, **kwargs)
        finally:
            owners.observation_timeout = previous

    monkeypatch.setattr(
        _producer_public_control, "retain_producer_completion", bounded_public_observation
    )
    access = SessionExportAccessContext(
        principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
    )
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(_ProducerSource(participant, CONTEXT),),
            producer_rules=()
            if automatic
            else (
                _ProducerMaintenanceRule(
                    HostProducerMaintenance(recovery=token, action="retain_completion"), CONTEXT
                ),
            ),
            producer_output_rules=(
                _ProducerOutputRule(
                    token,
                    CONTEXT,
                    access,
                    tuple(
                        _ProducerDisclosure(item.operation, access) for item in command.destinations
                    ),
                ),
            )
            if automatic
            else (),
            observation_timeout_s=30,
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
        await asyncio.sleep(0.1)
        observed = await host.aclose()
        assert observed.uncertain == 1 and observed.failed == 0
        assert len(provider.requests) == 1
    finally:
        release.set()
        async with asyncio.timeout(60):
            while (await host.aclose()).pending:
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                await asyncio.sleep(0.01)
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
    found = await application.lookup_producer_completion(command, context=CONTEXT)
    assert isinstance(found, ExactMatch)
    assert found.receipt.output.disposition == "answer"
    assert len(provider.requests) == 1
    assert host.inspect().uncertain == 0


async def test_pending_native_output_does_not_strand_maintenance(native_stores, monkeypatch):
    application, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores,
        with_exports=True,
        request_ttl_ms=900_000,
        provider_events=(
            (ModelStreamEvent.text_delta("Completed."), ModelStreamEvent.completed()),
        ),
    )
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    await application.register_producer_output(
        command, execution, context=resolver.recipient.context
    )
    authorize_execution(resolver)
    entered, release, observed_pending = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stream = provider.stream

    async def blocked_stream(request):
        async for event in stream(request):
            entered.set()
            await release.wait()
            yield event

    monkeypatch.setattr(provider, "stream", blocked_stream)
    receiver = application._request_coordinator._registration.receiving_owner
    read_output = receiver._read_producer_output

    async def observe_native(record):
        output = await read_output(record)
        if output is None:
            observed_pending.set()
        return output

    monkeypatch.setattr(receiver, "_read_producer_output", observe_native)

    async def execute():
        async for _ in application.execute_producer_output(
            command, execution, context=CONTEXT, producer_context=resolver.recipient.context
        ):
            pass

    execution_task = asyncio.create_task(execute())
    host = None
    try:
        await asyncio.wait_for(entered.wait(), 90)
        participant = command.admission.prepared.recipient
        page = await application.pending_producer_outputs(participant, context=CONTEXT)
        token = next(
            item.recovery for item in page.items if item.recovery.registration == command.operation
        )
        host = CollaborationHost(
            application,
            _HostRegistration(
                limits=HostOwnershipLimits(1, 1, 2, 262144),
                producer_sources=(_ProducerSource(participant, CONTEXT),),
                producer_rules=(
                    _ProducerMaintenanceRule(
                        HostProducerMaintenance(recovery=token, action="retain_completion"),
                        CONTEXT,
                    ),
                ),
                observation_timeout_s=30,
                shutdown_timeout_s=30,
            ),
        )
        async with asyncio.timeout(90):
            while not observed_pending.is_set():
                state = await host.service_once()
                assert state.failed == state.source_failures == 0
        closed = await host.aclose()
        assert not closed.pending
        assert closed.failed == closed.uncertain == 0
        current = await application.inspect_producer_output(command, context=CONTEXT)
        assert isinstance(current, ExactMatch)
        assert current.receipt.completion is None
        assert current.receipt.cleanup_ack is None
        assert not execution_task.done()
        assert len(provider.requests) == 1
    finally:
        release.set()
        await asyncio.wait_for(execution_task, 90)
        if host is not None:
            assert not (await host.aclose()).pending
    completion = await application.retain_producer_completion(command, context=CONTEXT)
    assert completion.output.disposition == "answer"
    assert len(provider.requests) == 1
