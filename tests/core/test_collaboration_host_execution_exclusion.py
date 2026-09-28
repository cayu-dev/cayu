"""Exact native no-start decisions free execution slots, not business responsibility."""

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import registration as participant_registration
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu import (
    CollaborationHost,
    ContinuationService,
    HostContinuationRule,
    HostOwnershipLimits,
    HostProducerExecution,
    HostProducerExecutionRule,
    HostProducerSource,
    HostRegistration,
    HostWaitRule,
)
from cayu.agents import AgentSpec
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._session_continuation import ContinuationRetirement
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.sessions import ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)

pytestmark = pytest.mark.anyio


def roomy_registration():
    original = participant_registration()
    return participant_registration(
        limits=original.bootstrap.limits.model_copy(
            update={"retained_bytes": 32 * 1024 * 1024, "operations": 2048, "events": 4096}
        )
    )


def registration(**rules):
    return HostRegistration(
        limits=HostOwnershipLimits(1, 1, 4, 524288),
        producer_sources=rules.pop("producer_sources", ()),
        producer_rules=(),
        observation_timeout_s=30,
        shutdown_timeout_s=30,
        **rules,
    )


async def pump(host, ready, *, reported=None):
    async with asyncio.timeout(180):
        while not ready():
            try:
                await host.service_once()
            except Exception as error:
                if reported is None:
                    raise
                reported.append(error)
            assert not host._source_errors, host._source_errors
            await asyncio.sleep(0.01)


async def park(app, initialized, accepted, participant, mandate, key, *, agent="root"):
    wait = wait_for(accepted, initialized).model_copy(
        update={"operation": initialized.operation("wait-" + key)}
    )
    request = RunRequest(agent_name=agent, messages=[Message.text("user", "Park " + key)])
    session, _ = await app.create_participant_session(
        ParticipantSessionCreationRequest(
            creation_key=initialized.owner.application_scope + ":" + key, request=request
        ),
        participant=participant,
        context=CONTEXT,
    )
    events = []
    async for event in app.execute_participant_session_to_wait(
        ParticipantSessionExecutionRequest(
            request=request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="park-" + key,
        ),
        wait,
        participant=participant,
        context=CONTEXT,
        wait_context=mandate,
    ):
        events.append(event)
    inventory, _ = await app.list_participant_sessions(participant, context=CONTEXT)
    reference = next(item for item in inventory if item.session_id == session.id)
    token = (await app.list_session_continuations(reference, context=CONTEXT)).items[0]
    record = await app.recover_session_continuation(token, context=CONTEXT)
    assert record.ticket.state == "WAITING", [
        e.payload.get("error") for e in events if "error" in e.payload
    ]
    return session, token


async def ready_rules(app, mandate, parked):
    waits = (await app.list_collaboration_waits(context=CONTEXT)).items
    host = CollaborationHost(
        app, registration(wait_rules=tuple(HostWaitRule(item.recovery, mandate) for item in waits))
    )
    records = None
    try:
        async with asyncio.timeout(180):
            while True:
                await host.service_once()
                assert not host._source_errors, host._source_errors
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
                records = [
                    await app.recover_session_continuation(token, context=CONTEXT)
                    for _, token in parked
                ]
                if all(record.latch is not None for record in records):
                    break
                await asyncio.sleep(0.01)
    finally:
        assert not (await host.aclose()).pending
    return tuple(
        HostContinuationRule(
            token,
            ResumeRequest(session_id=session.id, messages=[Message.text("user", "Continue")]),
            ContinuationService(
                ticket=record.ticket,
                latch=record.latch,
                continuation_id="continue-" + str(index),
                mode="inline",
                accepted_at=datetime.now(UTC).isoformat(),
            ),
            CONTEXT,
        )
        for index, ((session, token), record) in enumerate(zip(parked, records, strict=True))
    )


async def close_request(app, initialized, expected, context, key):
    current = await app.inspect_collaboration_request(expected, context=context)
    return await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation(key),
            expected=expected,
            expected_revision=current.revision,
            kind="cancel",
        ),
        context=context,
    )


@pytest.mark.parametrize("replay", [False, True])
async def test_excluded_continuation_frees_one_slot(native_stores, monkeypatch, replay):
    app, resolver, values = await public_setup(
        native_stores[0], reg=roomy_registration(), session_store=native_stores[1]
    )
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]] * 3
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"))
    parked_invocations = {}
    native_park = SessionContinuationOwner.park

    async def capture_park(owner, ticket, *, invocation):
        parked_invocations[ticket.session_id] = (owner, invocation)
        return await native_park(owner, ticket, invocation=invocation)

    monkeypatch.setattr(SessionContinuationOwner, "park", capture_park)
    parked = [
        await park(app, values[1], accepted, values[2].reference, resolver.context, key)
        for key in ("excluded", "successor")
    ]
    await close_request(app, values[1], accepted.expected, resolver.context, "close-wait-source")
    rules = await ready_rules(app, resolver.context, parked)
    first = parked[0][0].id
    primary = RuntimeError("native admission refused before claim")
    refused = asyncio.Event()
    from cayu.sessions.base import SessionStore

    claim = SessionStore._claim_continuation_admission

    async def refuse_claim(store, consumption):
        if consumption.ticket.session_id == first:
            refused.set()
            raise primary
        return await claim(store, consumption)

    monkeypatch.setattr(SessionStore, "_claim_continuation_admission", refuse_claim)
    from cayu.collaboration import _host_continuations as adapter

    ready = adapter.continuation_delivery_ready

    async def ordered_ready(app, expected, service, **kwargs):
        if expected.session.session_id != first and not refused.is_set():
            return False
        return await ready(app, expected, service, **kwargs)

    monkeypatch.setattr(adapter, "continuation_delivery_ready", ordered_ready)
    monkeypatch.setattr("cayu.collaboration._host.continuation_delivery_ready", ordered_ready)
    host = CollaborationHost(app, registration(continuation_rules=rules))
    reports = []
    try:
        await pump(host, lambda: host.inspect().failed == 1)
        original_failure = next(
            item.error for item in host._owned.inspect().completed if item.error is not None
        )
        assert refused.is_set() and len(provider.requests) == 2
        assert not host._owned.has_slot("execution")
        retained = await app.recover_session_continuation(rules[0].expected, context=CONTEXT)
        assert retained.consumption.receipt_stage == "prepared"
        assert not retained.consumption.admission_claimed
        owner, invocation = parked_invocations[first]
        excluded = await owner.exclude(
            ContinuationRetirement(
                ticket=retained.ticket,
                control_id="explicit-refusal",
                reason="failed",
                retired_at=datetime.now(UTC).isoformat(),
            ),
            invocation=invocation,
        )
        assert excluded.consumption.receipt_stage == "excluded"
        if replay:
            # Drain the first observer, then a fresh host observes exact native
            # excluded service replay rather than the failed turn's callback.
            await pump(host, lambda: host._owned.has_slot("execution"), reported=reports)
            assert not (await host.aclose()).pending
            host = CollaborationHost(app, registration(continuation_rules=rules))
        await pump(host, lambda: len(host._continuation_owner._finished) == 2, reported=reports)
        assert len(provider.requests) == 3
        assert len(reports) == 1
        assert reports[0] is original_failure
        assert host._owned.has_slot("execution")
        for _ in range(3):
            await host.service_once()
        assert len(provider.requests) == 3
        assert (
            await app.recover_session_continuation(rules[0].expected, context=CONTEXT) == excluded
        )
        other = await app.recover_session_continuation(rules[1].expected, context=CONTEXT)
        assert other.ticket.state == "CONSUMED"
        assert not (await host.aclose()).pending
        assert not (await host.aclose()).pending
    finally:
        await host.aclose()


@pytest.mark.parametrize("pending_cleanup", [False, True])
async def test_prelaunch_exclusion_frees_producer_slot(native_stores, monkeypatch, pending_cleanup):
    from tests.core import test_prepared_admission_public as fixture

    setup = fixture.setup

    async def roomy_setup(store):
        return await setup(store, reg=roomy_registration())

    monkeypatch.setattr(fixture, "setup", roomy_setup)
    app, resolver, admission, provider, _, initialized, command, execution = await output_scenario(
        native_stores,
        planned=True,
        request_ttl_ms=900_000,
        provider_events=tuple(
            (ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()) for _ in range(2)
        ),
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    authorize_execution(resolver)
    request = admission.expected.intent.request.model_copy(
        update={"operation": initialized.operation("independent-wait-source")}
    )
    accepted = await app.accept_collaboration_request(request, context=resolver.sender.context)
    parked = [
        await park(
            app,
            initialized,
            accepted,
            request.sender,
            resolver.sender.context,
            "unrelated",
            agent="reviewer",
        )
    ]
    await close_request(app, initialized, accepted.expected, resolver.sender.context, "close-other")
    rules = await ready_rules(app, resolver.sender.context, parked)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    page = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    entered, release = asyncio.Event(), asyncio.Event()
    failures = []
    execute = app.execute_producer_output

    async def blocked_execute(*args, **kwargs):
        entered.set()
        await release.wait()
        try:
            async for event in execute(*args, **kwargs):
                yield event
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(app, "execute_producer_output", blocked_execute)
    from cayu.collaboration import _host_continuations as adapter

    ready = adapter.continuation_delivery_ready

    async def after_producer(*args, **kwargs):
        return entered.is_set() and await ready(*args, **kwargs)

    monkeypatch.setattr(adapter, "continuation_delivery_ready", after_producer)
    monkeypatch.setattr("cayu.collaboration._host.continuation_delivery_ready", after_producer)
    host = CollaborationHost(
        app,
        registration(
            producer_sources=(HostProducerSource(admission.prepared.recipient, CONTEXT),),
            producer_execution_rules=(
                HostProducerExecutionRule(
                    HostProducerExecution(recovery=token), CONTEXT, resolver.recipient.context
                ),
            ),
            continuation_rules=rules,
        ),
    )
    reports = []
    receiver = app._request_coordinator._registration.receiving_owner
    acknowledge = receiver._acknowledge_producer_cleanup

    async def unavailable_ack(*args):
        raise ConnectionError("cleanup acknowledgement unavailable")

    try:
        await pump(host, entered.is_set)
        assert len(provider.requests) == 1 and not host._owned.has_slot("execution")
        if pending_cleanup:
            monkeypatch.setattr(receiver, "_acknowledge_producer_cleanup", unavailable_ack)
            with pytest.raises(CollaborationUnavailable):
                await close_request(
                    app, initialized, admission.expected, resolver.sender.context, "cancel-producer"
                )
        else:
            await close_request(
                app, initialized, admission.expected, resolver.sender.context, "cancel-producer"
            )
        before = await app.inspect_producer_output(command, context=CONTEXT)
        assert isinstance(before, ExactMatch) and before.receipt.state == "excluded"
        assert (before.receipt.cleanup_ack is None) == pending_cleanup
        release.set()
        await pump(host, lambda: len(host._continuation_owner._finished) == 1, reported=reports)
        assert len(failures) == len(reports) == 1
        assert reports[0] is failures[0]
        assert len(provider.requests) == 2  # park + unrelated continuation; no producer call
        assert host._owned.has_slot("execution")
        assert await app.inspect_producer_output(command, context=CONTEXT) == before
        assert not (await host.aclose()).pending
        assert not (await host.aclose()).pending
    finally:
        release.set()
        monkeypatch.setattr(receiver, "_acknowledge_producer_cleanup", acknowledge)
        await host.aclose()
    await app.settle_producer_output(command, context=CONTEXT)
