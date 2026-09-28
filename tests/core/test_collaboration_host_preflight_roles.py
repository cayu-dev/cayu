"""Read-only preparation failures release local turns, not native obligations."""

import asyncio

import pytest
from tests.core.test_collaboration_host_role_reconciliation import retry_preflight
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario
from tests.core.test_request_planning_public import complete_plan, scenario

from cayu import CollaborationHost, HostOwnershipLimits, HostRegistration
from cayu.collaboration._host import _PlanningRule, _ProducerRegistrationRule
from cayu.collaboration._host_planned_producer import HostPlannedProducer, _PlannedProducerRule

pytestmark = pytest.mark.anyio


async def test_continuation_preflight_failure_retries_same_host(native_stores, monkeypatch):
    from tests.core.test_collaboration_host_continuations import (
        test_host_discovers_and_recovers_public_park_after_reopen,
    )

    await test_host_discovers_and_recovers_public_park_after_reopen(
        native_stores,
        monkeypatch,
        cancel_observer=False,
        lost_ack=False,
        disable_before_reconcile=False,
        advance_before_reconcile=False,
        preflight_failure=True,
    )


@pytest.mark.parametrize("read_stage", ["outer", "owned"])
async def test_native_continuation_preflight_failure_frees_slot(
    native_stores, monkeypatch, read_stage, capsys, caplog
):
    import traceback
    import warnings
    from contextvars import ContextVar

    from tests.core.test_collaboration_host_execution_exclusion import (
        close_request,
        park,
        pump,
        ready_rules,
        registration,
        roomy_registration,
    )
    from tests.core.test_collaboration_request_foundation import public_setup

    from cayu.agents import AgentSpec
    from cayu.collaboration import _host_continuations as adapter
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner

    app, resolver, values = await public_setup(
        native_stores[0], reg=roomy_registration(), session_store=native_stores[1]
    )
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]] * 4
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"))
    parked = [
        await park(app, values[1], accepted, values[2].reference, resolver.context, key)
        for key in ("transient", "unrelated")
    ]
    await close_request(app, values[1], accepted.expected, resolver.context, "close-source")
    rules = await ready_rules(app, resolver.context, parked)
    first = parked[0][0].id
    entered = ContextVar("native_preflight", default=False)
    preparation_method = (
        "_prepare_service_observation" if read_stage == "outer" else "_read_service_record"
    )
    load_method = "load" if read_stage == "outer" else "load_continuation_ticket"
    prepare = getattr(SessionContinuationOwner, preparation_method)
    load = getattr(type(app.session_store), load_method)
    ready = adapter.continuation_delivery_ready
    primary = ExceptionGroup(
        "native read failures",
        [
            ConnectionError("native pre-admission read unavailable"),
            ExceptionGroup("read cleanup", [OSError("read cleanup failed")]),
        ],
    )
    secret = "continuation-read-private-canary"
    monkeypatch.setattr(app, "_secret_redactor", app._secret_redactor.with_secret(secret))
    primary.exceptions[0].args = ("native read " + secret,)
    primary.add_note(secret)
    failed = False
    cancelled_observer = None

    async def preparation(*args, **kwargs):
        token = entered.set(True)
        try:
            return await prepare(*args, **kwargs)
        finally:
            entered.reset(token)

    async def fail_read(store, session_id, **kwargs):
        nonlocal failed
        if entered.get() and session_id == first and not failed:
            failed = True
            if cancelled_observer is not None:

                def cancel_after_completion(_task):
                    cancelled_observer.cancel()
                    cancelled_observer.cancel()

                # The real receiving task finishes before cancellation is
                # delivered to its observer, exercising result-carried evidence.
                asyncio.current_task().add_done_callback(cancel_after_completion)
            raise primary
        return await load(store, session_id, **kwargs)

    async def ordered_ready(app, expected, service, **kwargs):
        if failed and len(provider.requests) == 2:
            retained = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
            assert retained.consumption is None
            assert retained.ticket.state == "WAITING"
        if expected.session.session_id == first:
            if failed and len(provider.requests) < 3:
                return False
        elif not failed:
            return False
        return await ready(app, expected, service, **kwargs)

    monkeypatch.setattr(SessionContinuationOwner, preparation_method, preparation)
    monkeypatch.setattr(type(app.session_store), load_method, fail_read)
    monkeypatch.setattr(adapter, "continuation_delivery_ready", ordered_ready)
    monkeypatch.setattr("cayu.collaboration._host.continuation_delivery_ready", ordered_ready)
    host = CollaborationHost(app, registration(continuation_rules=rules))
    reports = []
    try:
        if read_stage == "owned":
            from cayu.runtime._session_continuation import ContinuationUnavailable

            retained = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
            receiver = host._continuation_owner._receiver(retained)
            with pytest.raises(ContinuationUnavailable, match="dependency failed"):
                await receiver.service(
                    app, rules[0].request, rules[0].service, participant_context=rules[0].context
                )
            assert failed
            assert len(provider.requests) == 2
            retained = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
            assert retained.consumption is None
            failed = False
            cancelled_observer = asyncio.create_task(
                receiver.service(
                    app, rules[0].request, rules[0].service, participant_context=rules[0].context
                )
            )
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await cancelled_observer
            assert cancelled_observer.cancelled()
            assert cancelled_observer.cancelling() == 2
            diagnostic = cancelled.value.__cause__
            assert isinstance(diagnostic, ExceptionGroup)
            assert len(diagnostic.exceptions) == 2
            assert isinstance(diagnostic.exceptions[1], ExceptionGroup)
            assert len(diagnostic.exceptions[1].exceptions) == 1
            assert diagnostic is not primary  # public evidence uses the sanitized graph
            assert len(provider.requests) == 2
            retained = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
            assert retained.consumption is None
            assert retained.ticket.state == "WAITING"
            assert not receiver.owners.pending
            cancelled_observer = None
            failed = False
        with warnings.catch_warnings(record=True) as observed:
            await pump(host, lambda: len(host._continuation_owner._finished) == 2, reported=reports)
        assert len(reports) == 1 and reports[0] is not primary
        evidence = reports[0] if read_stage == "outer" else reports[0].__cause__
        assert isinstance(evidence, ExceptionGroup)
        assert len(evidence.exceptions) == 2
        assert isinstance(evidence.exceptions[1], ExceptionGroup)
        assert len(evidence.exceptions[1].exceptions) == 1
        streams = capsys.readouterr()
        assert secret not in (
            "".join(traceback.format_exception(reports[0]))
            + caplog.text
            + streams.out
            + streams.err
            + "".join(str(item.message) for item in observed)
        )
        assert len(provider.requests) == 4  # two parks, unrelated resume, exact retry
        assert host.inspect().failed == host.inspect().uncertain == 0
        assert host._owned.has_slot("execution")
        for _, token in parked:
            retained = await app.recover_session_continuation(token, context=CONTEXT)
            assert retained.ticket.state == "CONSUMED"
    finally:
        assert not (await host.aclose()).pending


@pytest.mark.parametrize("read_stage", ["host", "native"])
async def test_planning_preflight_failure_retries_same_host(
    native_stores, monkeypatch, read_stage, capsys, caplog
):
    import warnings
    from contextvars import ContextVar

    from tests.core.collaboration_preparation_assertions import (
        assert_safe_preparation_failure,
        private_preparation_failure,
    )

    from cayu.collaboration import _host_planning as adapter
    from cayu.collaboration import _planning_coordinator as native

    app, resolver, command, _, provider = await scenario(native_stores[0], "decline")
    from cayu.collaboration._clarification_state import clarification_commitment

    initialized = await app.initialize_collaboration()
    unrelated = await app.accept_collaboration_request(
        command.expected.intent.request.model_copy(
            update={"operation": initialized.operation("unrelated-preflight-request")}
        ),
        context=resolver.sender.context,
    )
    other = command.model_copy(
        update={
            "operation": initialized.operation("unrelated-preflight-plan"),
            "admission_operation": initialized.operation("unrelated-preflight-admission"),
            "expected": unrelated.expected,
            "expected_input_sha256": clarification_commitment(
                unrelated.expected, app._secret_redactor
            ),
        }
    )
    original = adapter.lookup_host_plan if read_stage == "host" else native.read_plan_in_transaction
    entered = ContextVar("native_planning_read", default=False)
    held = native._held

    async def held_attempt(*args, **kwargs):
        token = entered.set(not kwargs.get("read_only", False))
        try:
            return await held(*args, **kwargs)
        finally:
            entered.reset(token)

    primary = ConnectionError("planning read failed before dispatch")
    if read_stage == "native":
        primary, secret = private_preparation_failure(app._request_coordinator, monkeypatch)
    calls = 0
    finished = []
    service = adapter.service_application_plan

    async def track_service(*args, **kwargs):
        from cayu.collaboration._planning_records import RequestPlanningRecord

        result = await service(*args, **kwargs)
        if isinstance(result, RequestPlanningRecord):
            finished.append(result)
        return result

    monkeypatch.setattr(adapter, "service_application_plan", track_service)

    async def first_fails(*args, **kwargs):
        nonlocal calls
        if read_stage == "native" and not entered.get():
            return await original(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise primary
        return await original(*args, **kwargs)

    if read_stage == "host":
        monkeypatch.setattr(adapter, "lookup_host_plan", first_fails)
    else:
        monkeypatch.setattr(native, "_held", held_attempt)
        monkeypatch.setattr(native, "read_plan_in_transaction", first_fails)
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            planning_rules=tuple(
                _PlanningRule(selected, resolver.recipient.context) for selected in (command, other)
            ),
        ),
    )
    if read_stage == "host":
        await retry_preflight(host, primary, completed=lambda: len(finished) == 2)
    else:
        async with host, asyncio.timeout(180):
            with (
                warnings.catch_warnings(record=True) as observed,
                pytest.raises(Exception) as caught,
            ):
                await host.run()
            assert_safe_preparation_failure(caught.value, primary, secret, observed, capsys, caplog)
            assert host.inspect().uncertain == host.inspect().failed == 0
            assert host._owned.has_slot("maintenance")
            while len(finished) < 2:
                await host.service_once()
                await asyncio.sleep(0.001)
    assert calls >= 3 and not provider.requests
    for selected in (command, other):
        retained = await app.lookup_collaboration_plan(selected, context=resolver.recipient.context)
        assert retained.receipt.state == "declined"


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("read_stage", ["first", "last"])
async def test_attachment_preflight_failure_retries_same_host(
    native_stores, monkeypatch, planned, read_stage
):
    from cayu.collaboration import _host_planned_producer, _host_producer_registration

    plans = []

    async def keep_plan(app, command, context):
        plans.append(command)
        return await complete_plan(app, command, context)

    app, resolver, _, provider, _, _, command, _ = await output_scenario(
        native_stores,
        planned=True,
        planning_driver=keep_plan,
        with_exports=True,
        request_ttl_ms=900_000,
    )
    module = _host_planned_producer if planned else _host_producer_registration
    name = "lookup_host_plan" if planned else "producer_registration_ready"
    if read_stage == "last":
        if planned:
            from cayu.collaboration import _producer_preparation

            module, name = _producer_preparation, "prepare_producer_output"
        else:
            name = "_native_execution_commitment"
    original = getattr(module, name)
    primary = ConnectionError("attachment read failed before dispatch")
    calls = 0

    async def first_fails(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise primary
        return await original(*args, **kwargs)

    monkeypatch.setattr(module, name, first_fails)
    selected = HostPlannedProducer(
        plan=plans[0],
        operation=command.operation,
        binding_incarnation=command.binding_incarnation,
        execution_key=command.execution_key,
        limits=command.limits,
        destinations=command.destinations,
    )
    host = CollaborationHost(
        app,
        HostRegistration(
            limits=HostOwnershipLimits(1, 1, 2, 262144),
            producer_sources=(),
            producer_rules=(),
            planned_producer_rules=(
                _PlannedProducerRule(selected, CONTEXT, resolver.recipient.context),
            )
            if planned
            else (),
            producer_registration_rules=()
            if planned
            else (_ProducerRegistrationRule(command, CONTEXT, resolver.recipient.context),),
        ),
    )
    await retry_preflight(host, primary)
    assert calls >= 2 and not provider.requests
    assert await _host_producer_registration.producer_registration_ready(
        app, command, context=CONTEXT
    )
