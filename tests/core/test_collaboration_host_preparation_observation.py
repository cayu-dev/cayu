"""Nested native reads remain supervised past the public observation bound."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario
from tests.core.test_request_planning_public import complete_plan

from cayu.collaboration._host import CollaborationHost, _HostRegistration
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._host_planned_producer import HostPlannedProducer, _PlannedProducerRule

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "mode,occurrence", [("inspect", 1), ("lookup_admission", 1), ("lookup_admission", 2)]
)
async def test_planned_preparation_retains_nested_read(
    native_stores, monkeypatch, mode, occurrence
):
    plans = []

    async def retain_plan(app, command, context):
        plans.append(command)
        return await complete_plan(app, command, context)

    app, resolver, _, provider, _, _, command, _ = await output_scenario(
        native_stores,
        planned=True,
        planning_driver=retain_plan,
        with_exports=True,
        request_ttl_ms=900_000,
    )
    coordinator = app._request_coordinator
    owners = coordinator._owners
    original_run = coordinator._run
    store = native_stores[0]
    transaction = store._transaction
    selected = ContextVar("host_preparation_read", default=False)
    entered, release = asyncio.Event(), asyncio.Event()
    seen = 0
    blocked = False

    async def select_read(value, *, mode, **kwargs):
        nonlocal seen
        matches = mode == target_mode
        if matches:
            seen += 1
        token = selected.set(matches and seen == occurrence)
        try:
            return await original_run(value, mode=mode, **kwargs)
        finally:
            selected.reset(token)

    @asynccontextmanager
    async def blocked_read(scope, *, write):
        nonlocal blocked
        if selected.get() and not blocked:
            blocked = True
            # This is inside the native owner's dispatched task, not a fake
            # timeout outside its supervision boundary.
            monkeypatch.setattr(owners, "observation_timeout", 0.02)
            entered.set()
            await release.wait()
        async with transaction(scope, write=write) as tx:
            yield tx

    target_mode = mode
    monkeypatch.setattr(coordinator, "_run", select_read)
    monkeypatch.setattr(store, "_transaction", blocked_read)
    host = CollaborationHost(
        app,
        _HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            planned_producer_rules=(
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
            ),
            observation_timeout_s=60,
            shutdown_timeout_s=0.01,
        ),
    )
    observer = asyncio.create_task(host.run())
    try:
        await asyncio.wait_for(entered.wait(), 90)
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        await asyncio.sleep(0.06)
        for _ in range(4):
            state = await host.service_once()
            assert state.uncertain == 1 and state.failed == 0
        assert seen == occurrence
        assert (await host.aclose()).pending
        assert not provider.requests
    finally:
        monkeypatch.setattr(owners, "observation_timeout", 60)
        release.set()
        if not observer.done():
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        async with asyncio.timeout(90):
            while (await host.aclose()).pending:
                await asyncio.sleep(0.01)
        await app.drain_collaboration_requests()
    assert host.inspect().failed == 0
    assert not provider.requests
