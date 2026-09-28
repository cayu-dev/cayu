"""A receiving commit can be recovered without repeating its external effect."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from tests.core.test_collaboration_host_execution_exclusion import (
    close_request,
    park,
    pump,
    registration,
    roomy_registration,
)
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_budget_refusal import authorize_execution
from tests.core.test_producer_output_contracts import output_scenario

from cayu import (
    CollaborationHost,
    ContinuationConflict,
    ContinuationService,
    HostContinuationRule,
    HostProducerExecution,
    HostProducerExecutionRule,
    HostProducerSource,
    HostWaitRule,
)
from cayu.agents import AgentSpec
from cayu.collaboration import _host_producer_registration as attachment_adapter
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._wait_coordinator import _elected_latch
from cayu.collaboration._wait_discovery import resolve_wait
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _producer_output_store
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.sessions import ResumeRequest

pytestmark = pytest.mark.anyio


async def test_execution_attachment_ack_loss_recovers_before_dispatch(native_stores, monkeypatch):
    app, resolver, admission, provider, _, _, command, execution = await output_scenario(
        native_stores,
        planned=True,
        with_exports=True,
        request_ttl_ms=900_000,
        provider_events=((ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()),),
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    authorize_execution(resolver)
    attach = _producer_output_store.attach_native_producer
    calls = 0

    async def lose_attachment_ack(store, record):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("source committed; attachment not dispatched")
        result = await attach(store, record)
        if calls == 2:
            raise ConnectionError("attachment committed; acknowledgement lost")
        return result

    monkeypatch.setattr(_producer_output_store, "attach_native_producer", lose_attachment_ack)
    with pytest.raises(CollaborationUnavailable):
        await app.register_producer_output(command, execution, context=resolver.recipient.context)
    assert await native_stores[1]._read_native_producer_attachment(command) is None
    token = (
        (await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT))
        .items[0]
        .recovery
    )
    ready = attachment_adapter.producer_registration_ready
    immediate_read_failed = False

    async def fail_immediate_read(*args, **kwargs):
        nonlocal immediate_read_failed
        if calls == 2 and not immediate_read_failed:
            immediate_read_failed = True
            raise OSError("immediate attachment read unavailable")
        return await ready(*args, **kwargs)

    monkeypatch.setattr(attachment_adapter, "producer_registration_ready", fail_immediate_read)
    host = CollaborationHost(
        app,
        registration(
            producer_sources=(HostProducerSource(admission.prepared.recipient, CONTEXT),),
            producer_execution_rules=(
                HostProducerExecutionRule(
                    HostProducerExecution(recovery=token), CONTEXT, resolver.recipient.context
                ),
            ),
        ),
    )
    try:
        await pump(host, lambda: host.inspect().failed == 1)
        failure = next(
            item.error for item in host._owned.inspect().completed if item.error is not None
        )
        assert isinstance(failure, ExceptionGroup) and immediate_read_failed
        assert not provider.requests and not host._owned.has_slot("execution")
        reports = []
        await pump(host, lambda: bool(reports), reported=reports)
        assert reports == [failure]
        assert host._owned.has_slot("execution")
        await pump(host, lambda: len(provider.requests) == 1 and host._owned.has_slot("execution"))
        assert calls == 2
        assert len(provider.requests) == 1
        assert not (await host.aclose()).pending
    finally:
        await host.aclose()


@pytest.mark.parametrize("close_during_recovery", [False, True])
async def test_latch_commit_source_ack_failure_recovers_same_host(
    native_stores, monkeypatch, close_during_recovery
):
    app, resolver, values = await public_setup(
        native_stores[0], session_store=native_stores[1], reg=roomy_registration()
    )
    accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()] for _ in range(4)]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="root", model="model"))
    parked = [
        await park(app, values[1], accepted, values[2].reference, resolver.context, key)
        for key in ("ack-failure", "independent-wait")
    ]
    await close_request(app, values[1], accepted.expected, resolver.context, "close-source")
    waits = (await app.list_collaboration_waits(context=CONTEXT)).items
    coordinator = app._wait_coordinator
    selected = []
    continuations = []
    retained_waits = {}
    for item in waits:
        found = await resolve_wait(coordinator, item.recovery, context=resolver.context)
        retained_waits[found.receipt.delivery_ticket.session_id] = (item.recovery, found.receipt)
    for session, token in parked:
        record = await app.recover_session_continuation(token, context=CONTEXT)
        recovery, wait = retained_waits[session.id]
        snapshot = await coordinator._observe_owned(wait, context=resolver.context)
        latch = _elected_latch(snapshot, app._secret_redactor)
        selected.append(HostWaitRule(recovery, resolver.context))
        continuations.append(
            HostContinuationRule(
                token,
                ResumeRequest(session_id=session.id, messages=[Message.text("user", "Continue")]),
                ContinuationService(
                    ticket=record.ticket,
                    latch=latch,
                    continuation_id="continue",
                    mode="inline",
                    accepted_at=datetime.now(UTC).isoformat(),
                ),
                CONTEXT,
            )
        )
    store = native_stores[0]
    acknowledge = store.record_wait_delivery
    primary = OSError("source acknowledgement write failed before commit")
    rejected = False
    native_latches = 0
    latch_write = native_stores[1].latch_continuation

    async def count_latches(*args, **kwargs):
        nonlocal native_latches
        result = await latch_write(*args, **kwargs)
        native_latches += 1
        return result

    # Patch the qualified class, not an unqualified per-instance store override.
    session_type = type(native_stores[1])

    async def class_latch(self, *args, **kwargs):
        return await count_latches(*args, **kwargs)

    monkeypatch.setattr(session_type, "latch_continuation", class_latch)

    async def fail_source_ack(initialized, wait, **kwargs):
        nonlocal rejected
        if not rejected:
            rejected = True
            retained = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
            assert retained.latch is not None
            raise primary
        return await acknowledge(initialized, wait, **kwargs)

    monkeypatch.setattr(store, "record_wait_delivery", fail_source_ack)
    inspect_latch = SessionContinuationOwner._inspect_latch_owned
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_inspection(owner, candidate):
        retained = await inspect_latch(owner, candidate)
        entered.set()
        await release.wait()
        return retained

    configured = replace(
        registration(wait_rules=tuple(selected), continuation_rules=tuple(continuations)),
        shutdown_timeout_s=0.01 if close_during_recovery else 30,
    )
    host = CollaborationHost(
        app, replace(configured, continuation_rules=()) if close_during_recovery else configured
    )
    # Recovery is not permission to publish a missing latch, even for an elected
    # wait with a genuine ticket. The public host below owns the first delivery.
    first_wait = retained_waits[parked[0][0].id][1]
    before = await coordinator.inspect(first_wait, context=resolver.context)
    assert (
        await coordinator._reconcile_delivery_owned(
            first_wait,
            context=resolver.context,
            continuation_owner=host._wait_owner._receiver(first_wait),
        )
        is None
    )
    assert await coordinator.inspect(first_wait, context=resolver.context) == before
    assert native_latches == 0
    if close_during_recovery:
        # A genuine native owner may accept the current WAITING ticket instead
        # of the source registration's earlier ARMING representation. Identity
        # excludes lifecycle state/revision, but the retained receipt bytes do not.
        current = await app.recover_session_continuation(parked[0][1], context=CONTEXT)
        advanced = continuations[0].service.latch.model_copy(update={"ticket": current.ticket})
        assert advanced.ticket != first_wait.delivery_ticket
        retained = await host._wait_owner._receiver(first_wait).latch(advanced)
        assert retained.latch == advanced
        forged = continuations[0].service.model_copy(
            update={"latch": advanced.model_copy(update={"outcome_digest": "different-result"})}
        )
        with pytest.raises(ContinuationConflict):
            await host._wait_owner._receiver(first_wait).service(
                app, continuations[0].request, forged, participant_context=CONTEXT
            )
        assert await app.recover_session_continuation(parked[0][1], context=CONTEXT) == retained
        assert len(provider.requests) == 2
    monkeypatch.setattr(SessionContinuationOwner, "_inspect_latch_owned", blocked_inspection)
    observer = None
    try:
        observer = asyncio.create_task(host.run())
        await asyncio.wait_for(entered.wait(), 180)
        assert not host._owned.has_slot("maintenance")
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        assert not host._owned.has_slot("maintenance")
        if close_during_recovery:
            assert (await host.aclose()).pending
        release.set()
        reports = []
        if close_during_recovery:
            async with asyncio.timeout(180):
                while True:
                    try:
                        if not (await host.aclose()).pending:
                            break
                    except Exception as error:
                        if error is not primary:
                            raise
                        reports.append(error)
                    await asyncio.sleep(0.01)
            assert reports == [primary]
            repaired = await coordinator.inspect(first_wait, context=resolver.context)
            assert repaired.delivery == "accepted" and not repaired.source_pins
            assert (
                repaired.delivery_receipt_digest
                == sha256(contract_bytes(advanced, redactor=app._secret_redactor)).hexdigest()
            )
            assert native_latches == 1  # Closing cannot deliver the unrelated wait.
            host = CollaborationHost(app, configured)

        def completed():
            for outcome in host._owned.inspect().completed:
                if outcome.error is not None and outcome.error is not primary:
                    raise outcome.error
            for error in reports:
                if error is not primary:
                    raise error
            return (
                len(host._wait_owner._finished) == 2
                and len(host._continuation_owner._finished) == 2
            )

        await pump(host, completed, reported=reports)
        assert reports == [primary]
        assert native_latches == 2 and len(provider.requests) == 4
        for _, token in parked:
            record = await app.recover_session_continuation(token, context=CONTEXT)
            assert record.ticket.state == "CONSUMED"
            snapshot = await coordinator.inspect(
                retained_waits[record.ticket.session_id][1], context=resolver.context
            )
            assert snapshot.delivery == "accepted" and not snapshot.source_pins
            assert (
                snapshot.delivery_receipt_digest
                == sha256(contract_bytes(record.latch, redactor=app._secret_redactor)).hexdigest()
            )
            # Ordinary public delivery must replay the exact repaired receipt,
            # even after continuation consumed the ticket and advanced its state.
            wait = retained_waits[record.ticket.session_id][1]
            replay = await app.deliver_collaboration_wait(
                wait,
                context=resolver.context,
                continuation_owner=host._wait_owner._receiver(wait),
            )
            assert replay == snapshot
        assert native_latches == 2 and len(provider.requests) == 4
        assert host._owned.has_slot("maintenance") and host._owned.has_slot("execution")
        assert not (await host.aclose()).pending
    finally:
        release.set()
        if observer is not None and not observer.done():
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
        await host.aclose()
