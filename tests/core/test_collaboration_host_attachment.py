"""Source registration does not stand in for native attachment completion."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
from cayu.collaboration._host import (
    CollaborationHost,
    _HostRegistration,
    _ProducerRegistrationRule,
)
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_producer_registration import producer_registration_ready
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime import _producer_output_store

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("boundary", ["before", "source-ack-loss", "after", "cancelled-observer"])
async def test_host_registration_requires_both_owners(
    native_stores, monkeypatch, boundary, planned_host=False
):
    from tests.core.test_request_planning_public import complete_plan

    plans = []

    async def retain_plan(application, command, context):
        plans.append(command)
        return await complete_plan(application, command, context)

    application, resolver, _, provider, _, _, command, execution = await output_scenario(
        native_stores,
        planned=True,
        planning_driver=retain_plan,
        with_exports=True,
        request_ttl_ms=900_000,
    )
    monkeypatch.setattr(application._request_coordinator._owners, "observation_timeout", 60)
    sessions = native_stores[1]
    attach = _producer_output_store.attach_native_producer
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def at_boundary(store, record):
        nonlocal calls
        calls += 1
        if boundary in {"before", "source-ack-loss"} and calls == 1:
            raise ConnectionError("source committed before native attachment")
        result = await attach(store, record)
        if calls == 1:
            if boundary == "after":
                raise ConnectionError("native attachment acknowledgement lost")
            if boundary == "cancelled-observer":
                entered.set()
                await release.wait()
        return result

    monkeypatch.setattr(_producer_output_store, "attach_native_producer", at_boundary)
    if boundary == "before":
        with pytest.raises(CollaborationUnavailable):
            await application.register_producer_output(
                command, execution, context=resolver.recipient.context
            )
        assert isinstance(
            await application.lookup_producer_registration(command, context=CONTEXT), ExactMatch
        )
        assert await sessions._read_native_producer_attachment(command) is None
        assert not await producer_registration_ready(application, command, context=CONTEXT)

    planned_rules = ()
    if planned_host:
        from cayu.collaboration._host_planned_producer import (
            HostPlannedProducer,
            _PlannedProducerRule,
        )

        planned_rules = (
            _PlannedProducerRule(
                HostPlannedProducer(
                    plan=plans[0],
                    operation=command.operation,
                    binding_incarnation=command.binding_incarnation,
                    execution_key=command.execution_key,
                    limits=command.limits,
                    destinations=command.destinations,
                ),
                CONTEXT,
                resolver.recipient.context,
            ),
        )
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            producer_registration_rules=()
            if planned_host
            else (_ProducerRegistrationRule(command, CONTEXT, resolver.recipient.context),),
            planned_producer_rules=planned_rules,
            observation_timeout_s=60,
            shutdown_timeout_s=0.01 if boundary == "cancelled-observer" else 30,
        ),
    )
    observer = None
    errors = []
    try:
        if boundary == "cancelled-observer":
            observer = asyncio.create_task(host.run())
            await asyncio.wait_for(entered.wait(), 90)
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
            assert (await host.aclose()).uncertain == 1
            assert host.inspect().failed == 0
            assert await producer_registration_ready(application, command, context=CONTEXT)
            release.set()
        else:
            async with asyncio.timeout(120):
                while True:
                    try:
                        state = await host.service_once()
                    except Exception as error:
                        errors.append(error)
                        break
                    for result in host._owned.inspect().completed:
                        if result.error is not None and not result.reconciled:
                            raise result.error
                    assert state.source_failures == 0, host._source_errors
                    if state.serviced:
                        break
            assert len(errors) == (1 if boundary in {"source-ack-loss", "after"} else 0)
            if planned_host:
                assert len(host._attached_planned) == 1

                async def redundant_preparation(*args, **kwargs):
                    pytest.fail("An exactly attached plan was prepared again by the same host")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        "cayu.collaboration._host_planned_producer.lookup_host_plan",
                        redundant_preparation,
                    )
                    for _ in range(3):
                        await host.service_once()
                    assert not host._owned.pending
            await host.service_once()
        async with asyncio.timeout(30):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.001)
        assert host.inspect().failed == 0
        assert not provider.requests
        assert await producer_registration_ready(application, command, context=CONTEXT)
        retained = await sessions._read_native_producer_attachment(command)
        assert retained.command == command
        assert calls == (2 if boundary in {"before", "source-ack-loss"} else 1)
        # Exact attachment readback must not bless another immutable authority.
        with pytest.raises(CollaborationConflict):
            await sessions._read_native_producer_attachment(
                command.model_copy(update={"binding_incarnation": "replacement"})
            )
    finally:
        release.set()
        if observer is not None and not observer.done():
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        await host.aclose()
        await application.drain_collaboration_requests()


async def test_planned_host_repairs_source_only_registration(native_stores, monkeypatch):
    await test_host_registration_requires_both_owners(
        native_stores, monkeypatch, "before", planned_host=True
    )


async def test_planned_attachment_lost_ack_reconciles_without_reattachment(
    native_stores, monkeypatch
):
    from tests.core.test_request_planning_public import complete_plan

    from cayu.collaboration import _host_planned_producer
    from cayu.collaboration._host_planned_producer import (
        HostPlannedProducer,
        _PlannedProducerRule,
    )

    plans = []

    async def retain_plan(application, command, context):
        plans.append(command)
        return await complete_plan(application, command, context)

    application, resolver, _, provider, _, _, command, _ = await output_scenario(
        native_stores,
        planned=True,
        planning_driver=retain_plan,
        with_exports=True,
        request_ttl_ms=900_000,
    )
    primary = ExceptionGroup("attachment acknowledgement lost", [ConnectionError("lost ack")])
    attach = _host_planned_producer.attach_host_producer
    calls = []

    async def after_attachment(*args, **kwargs):
        await attach(*args, **kwargs)
        calls.append(True)
        raise primary

    monkeypatch.setattr(_host_planned_producer, "attach_host_producer", after_attachment)
    selected = HostPlannedProducer(
        plan=plans[0],
        operation=command.operation,
        binding_incarnation=command.binding_incarnation,
        execution_key=command.execution_key,
        limits=command.limits,
        destinations=command.destinations,
    )
    host = CollaborationHost(
        application,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            planned_producer_rules=(
                _PlannedProducerRule(selected, CONTEXT, resolver.recipient.context),
            ),
            observation_timeout_s=30,
            shutdown_timeout_s=30,
        ),
    )
    async with host, asyncio.timeout(180):
        with pytest.raises(ExceptionGroup) as caught:
            await host.run()
        assert caught.value is primary
        assert calls == [True]
        assert not host.inspect().failed
        assert len(host._attached_planned) == 1
        assert await producer_registration_ready(application, command, context=CONTEXT)
    assert not host.inspect().pending
    assert not provider.requests
