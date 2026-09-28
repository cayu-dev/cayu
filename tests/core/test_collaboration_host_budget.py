"""Independent host slots still compete under one native monetary ceiling."""

import asyncio
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.core import test_prepared_admission_public as preparation
from tests.core.test_budget_binding import _binding, _limit
from tests.core.test_participant_identity import CONTEXT, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_budget_refusal import refusal_ledger as refusal_ledger
from tests.core.test_producer_output_contracts import output_scenario

from cayu import (
    CollaborationHost,
    HostOwnershipLimits,
    HostProducerExecution,
    HostProducerExecutionRule,
    HostProducerSource,
    HostRegistration,
    SessionExportAccessContext,
)
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent


@pytest.mark.anyio
async def test_independent_hosts_preserve_shared_budget_and_settlement(
    native_stores, refusal_ledger, monkeypatch
):
    registered = registration()
    registered = replace(
        registered,
        bootstrap=registered.bootstrap.model_copy(
            update={
                "limits": registered.bootstrap.limits.model_copy(
                    update={"retained_bytes": 9 * 1024 * 1024, "events": 512}
                )
            }
        ),
    )
    setup = preparation.setup

    async def same_namespace(store, **kwargs):
        return await setup(store, reg=registered, **kwargs)

    monkeypatch.setattr(preparation, "setup", same_namespace)
    root = "host-shared-root:" + uuid4().hex

    def binding(scope):
        return _binding(
            application_scope=scope,
            binding_id=root,
            root_budget_id=root,
            limits=(
                _limit().model_copy(
                    update={"key": root, "max_estimated_cost": Decimal("0.000003")}
                ),
            ),
        )

    values = []
    hosts = []
    release, entered = asyncio.Event(), asyncio.Event()
    first_running = None

    async def check(host):
        state = await host.service_once()
        if host._source_errors:
            raise ExceptionGroup("Host discovery failed", list(host._source_errors.values()))
        for outcome in host._owned.inspect().completed:
            if outcome.error is not None:
                raise outcome.error
        return state

    try:
        for index in range(2):
            scenario = await output_scenario(
                native_stores,
                with_exports=True,
                planned=True,
                operation_prefix=f"host-{index}:",
                request_ttl_ms=900_000,
                budget_ledger=refusal_ledger[0],
                budget_binding_factory=binding,
                provider_events=(
                    (
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ),
                ),
            )
            values.append(scenario)
            app, resolver, _, provider, _, _, command, execution = scenario
            monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
            await app.register_producer_output(
                command, execution, context=resolver.recipient.context
            )
            authorize_execution(resolver)
            pending = await app.pending_producer_outputs(
                command.admission.prepared.recipient, context=CONTEXT
            )
            token = next(
                item.recovery
                for item in pending.items
                if item.recovery.registration == command.operation
            )
            hosts.append(
                CollaborationHost(
                    app,
                    HostRegistration(
                        limits=HostOwnershipLimits(1, 1, 4, 262144),
                        producer_sources=(
                            HostProducerSource(command.admission.prepared.recipient, CONTEXT),
                        ),
                        producer_rules=(),
                        producer_execution_rules=(
                            HostProducerExecutionRule(
                                HostProducerExecution(recovery=token),
                                CONTEXT,
                                resolver.recipient.context,
                            ),
                        ),
                        observation_timeout_s=30,
                        shutdown_timeout_s=30,
                    ),
                )
            )
        first_provider = values[0][3]
        stream = first_provider.stream

        async def blocked(request):
            entered.set()
            await release.wait()
            async for event in stream(request):
                yield event

        monkeypatch.setattr(first_provider, "stream", blocked)
        first_running = asyncio.create_task(hosts[0].run())
        async with asyncio.timeout(180):
            while not entered.is_set():
                if first_running.done():
                    await first_running
                await asyncio.sleep(0.01)
            first_rows = await refusal_ledger[0]._scan_reservation_records(
                session_id=values[0][4].id
            )
            assert len(first_rows) == 1
            assert first_rows[0].reserved_amount == Decimal("0.000002")
            while not (await check(hosts[1])).serviced:
                await asyncio.sleep(0.01)
        second_app, _, _, second_provider, second_session, *_ = values[1]
        events = await second_app.session_store.load_events(second_session.id)
        assert any(event.type is EventType.BUDGET_RESERVATION_FAILED for event in events)
        assert not second_provider.requests
        assert not await refusal_ledger[0]._scan_reservation_records(session_id=second_session.id)
        assert hosts[0].inspect().pending
        release.set()
        await hosts[0].aclose()
        await first_running
        assert len(first_provider.requests) == 1
        for host, scenario in zip(hosts, values, strict=True):
            async with asyncio.timeout(90):
                while (await host.aclose()).pending:
                    await asyncio.sleep(0.01)
            app, resolver, _, provider, _, _, command, _ = scenario
            completion = await app.retain_producer_completion(command, context=CONTEXT)
            assert completion.output.disposition != "answer"
            disclosure = SessionExportAccessContext(
                principal=resolver.recipient.context.principal, mandate=resolver.recipient.context
            )
            elected = await app.publish_producer_outcome(command, context=disclosure)
            assert elected.command.outcome == "failed"
            finalized = await app.settle_producer_output(command, context=CONTEXT)
            assert await app.settle_producer_output(command, context=CONTEXT) == finalized
            assert len(provider.requests) == (1 if host is hosts[0] else 0)
        final_rows = await refusal_ledger[0]._scan_reservation_records(session_id=values[0][4].id)
        assert len(final_rows) == 1
        assert final_rows[0].status == "reconciled"
        assert final_rows[0].reservation_id == first_rows[0].reservation_id
    finally:
        release.set()
        for host in hosts:
            await host.aclose()
        if first_running is not None and not first_running.done():
            first_running.cancel()
            await asyncio.gather(first_running, return_exceptions=True)
        for scenario in values:
            await scenario[0].drain_collaboration_requests()
