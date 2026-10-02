"""Native discovery recovers a real parked invocation without its local handle."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ContinuationConflict
from cayu.agents import AgentSpec
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("cancel_observer", "lost_ack", "disable_before_reconcile", "advance_before_reconcile"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        pytest.param(False, True, True, False, id="disabled-ackloss"),
        pytest.param(False, True, False, True, id="successor-ackloss"),
    ],
)
async def test_host_discovers_and_recovers_public_park_after_reopen(
    native_stores,
    monkeypatch,
    cancel_observer,
    lost_ack,
    disable_before_reconcile,
    advance_before_reconcile,
    preflight_failure=False,
):
    application, resolver, values = await public_setup(
        native_stores[0], session_store=native_stores[1]
    )
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1])
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("Ready to wait."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("Wait result received."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("Independent successor completed."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("Another independent successor completed."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="root", model="model"))
    creation = ParticipantSessionCreationRequest(
        creation_key="host-continuation-" + uuid4().hex,
        request=RunRequest(agent_name="root", messages=[Message.text("user", "Wait for review.")]),
    )
    participant = values[2].reference
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    inventory, _cursor = await application.list_participant_sessions(
        participant, context=CONTEXT, limit=1
    )
    reference = inventory[0]
    assert (await application.list_session_continuations(reference, context=CONTEXT)).items == ()
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="park",
    )
    async for _ in application.execute_participant_session_to_wait(
        execution, wait, participant=participant, context=CONTEXT, wait_context=resolver.context
    ):
        pass
    assert len(provider.requests) == 1
    page = await application.list_session_continuations(reference, context=CONTEXT, limit=1)
    assert len(page.items) == 1 and page.next_cursor is not None
    token = page.items[0]
    backend, address = native_stores[3]
    reopened = native_stores[1]
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        reopened = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
    elif backend == "postgres":
        from cayu.storage.postgres import PostgresSessionStore

        reopened = PostgresSessionStore(address)
    other = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
        session_store=reopened,
    )
    try:
        await other.initialize_collaboration()
        recovered = await other.recover_session_continuation(token, context=CONTEXT)
        assert recovered.ticket.state == "WAITING"
        assert recovered.ticket.session_instance_id == session.instance_id
        assert (await other.list_session_continuations(reference, context=CONTEXT, limit=1)) == page
        end = await other.list_session_continuations(
            reference, context=CONTEXT, after=page.next_cursor, limit=1
        )
        assert end.items == () and end.next_cursor is None
        with pytest.raises(ContinuationConflict):
            await other.recover_session_continuation(
                token.model_copy(update={"preparation_digest": "0" * 64}), context=CONTEXT
            )
        with pytest.raises(CollaborationAccessDenied):
            await other.recover_session_continuation(
                token, context=CollaborationAccessContext(principal="foreign")
            )
        assert len(provider.requests) == 1
        # The replacement host receives the election through the native latch
        # owner; discovery alone does not execute the parked invocation.
        from cayu import CollaborationHost, HostOwnershipLimits, HostRegistration, HostWaitRule
        from cayu.collaboration.requests import RequestControl

        await application.control_collaboration_request(
            RequestControl(
                operation=values[1].operation("host-source-close"),
                expected=receipt.expected,
                expected_revision=1,
                kind="cancel",
            ),
            context=resolver.context,
        )
        waits = await other.list_collaboration_waits(context=CONTEXT)
        assert len(waits.items) == 1
        observations = []
        observe = other._wait_coordinator._observe_owned

        async def counted_observation(*args, **kwargs):
            observations.append(True)
            return await observe(*args, **kwargs)

        monkeypatch.setattr(other._wait_coordinator, "_observe_owned", counted_observation)
        host = CollaborationHost(
            other,
            HostRegistration(
                limits=HostOwnershipLimits(1, 1, 4, 262144),
                producer_sources=(),
                producer_rules=(),
                wait_rules=(HostWaitRule(waits.items[0].recovery, resolver.context),),
                observation_timeout_s=30,
                shutdown_timeout_s=30,
            ),
        )
        async with host:
            if cancel_observer:
                from cayu.collaboration._wait_coordinator import CollaborationWaitLatchReceiver

                latch_entered, latch_release = asyncio.Event(), asyncio.Event()
                latch_calls = []
                authenticate = CollaborationWaitLatchReceiver._authenticate_latch
                source_store, _initialized = other._participant_coordinator._ready()
                source_timeout = source_store._owners.observation_timeout

                async def blocked_latch_authentication(receiver, latch):
                    result = await authenticate(receiver, latch)
                    if receiver._store is source_store:
                        latch_calls.append(True)
                        latch_entered.set()
                        await latch_release.wait()
                    return result

                monkeypatch.setattr(
                    CollaborationWaitLatchReceiver,
                    "_authenticate_latch",
                    blocked_latch_authentication,
                )
                source_store._owners.observation_timeout = 0.02
                latch_observer = asyncio.create_task(host.run())
                try:
                    await asyncio.wait_for(latch_entered.wait(), 30)
                    latch_observer.cancel()
                    assert latch_observer.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await latch_observer
                    assert latch_observer.cancelled() and latch_observer.cancelling() == 1
                    await asyncio.sleep(0.06)
                    state = await host.service_once()
                    assert state.uncertain == 1 and state.failed == 0
                    assert latch_calls == [True]
                finally:
                    source_store._owners.observation_timeout = source_timeout
                    latch_release.set()
                    if not latch_observer.done():
                        latch_observer.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await latch_observer
            async with asyncio.timeout(60):
                while True:
                    await host.service_once()
                    for outcome in host._owned.inspect().completed:
                        if outcome.error is not None:
                            raise outcome.error
                    retained = await other.recover_session_continuation(token, context=CONTEXT)
                    if retained.latch is not None:
                        break
                    await asyncio.sleep(0.01)
                while (await host.service_once()).pending:
                    await asyncio.sleep(0.01)
                settled_observations = len(observations)
                assert settled_observations > 0
                for _ in range(3):
                    assert not (await host.service_once()).pending
                assert len(observations) == settled_observations
        assert host.inspect().uncertain == 0
        assert host.inspect().failed == 0
        assert retained.ticket.state == "WAITING"
        assert len(provider.requests) == 1
        from cayu import ContinuationService, HostContinuationRule
        from cayu.sessions import ResumeRequest

        other.register_provider(provider, default=True)
        other.register_agent(AgentSpec(name="root", model="model"))
        rule = HostContinuationRule(
            token,
            ResumeRequest(
                session_id=session.id, messages=[Message.text("user", "Continue after the wait.")]
            ),
            ContinuationService(
                ticket=retained.ticket,
                latch=retained.latch,
                continuation_id="host-wait-continuation",
                mode="inline",
                accepted_at=datetime.now(UTC).isoformat(),
            ),
            context=CONTEXT,
        )
        continuation_host = CollaborationHost(
            other,
            HostRegistration(
                limits=HostOwnershipLimits(1, 1, 4, 262144),
                producer_sources=(),
                producer_rules=(),
                continuation_rules=(rule,),
                observation_timeout_s=30,
                shutdown_timeout_s=0.01 if cancel_observer or lost_ack else 30,
            ),
        )
        if preflight_failure:
            from cayu.collaboration import _host_continuations as adapter

            recover = adapter.recover_session_continuation
            primary = ConnectionError("continuation read failed before admission")
            reads = 0

            async def first_fails(*args, **kwargs):
                nonlocal reads
                reads += 1
                if reads == 1:
                    raise primary
                return await recover(*args, **kwargs)

            monkeypatch.setattr(adapter, "recover_session_continuation", first_fails)
            async with asyncio.timeout(60):
                with pytest.raises(ConnectionError) as caught:
                    await continuation_host.run()
            assert caught.value is primary
            assert continuation_host.inspect().uncertain == continuation_host.inspect().failed == 0
            assert len(provider.requests) == 1
        if lost_ack:
            from cayu.collaboration import _host_continuation_recovery
            from cayu.runtime._session_continuation_owner import SessionContinuationOwner

            original_service = SessionContinuationOwner._service_owned
            original_release = _host_continuation_recovery.read_continuation_release
            primary = ExceptionGroup("continuation acknowledgement lost", [OSError("lost ack")])
            unavailable = False
            dispatches = []

            async def committed_then_failed(*args, **kwargs):
                nonlocal unavailable
                result = await original_service(*args, **kwargs)
                assert result.dispatched
                dispatches.append(result.record)
                unavailable = True
                raise primary

            async def unavailable_release(*args, **kwargs):
                if unavailable:
                    return None
                return await original_release(*args, **kwargs)

            monkeypatch.setattr(SessionContinuationOwner, "_service_owned", committed_then_failed)
            monkeypatch.setattr(
                _host_continuation_recovery, "read_continuation_release", unavailable_release
            )
            failures = []
            try:
                async with asyncio.timeout(90):
                    while not continuation_host.inspect().failed:
                        await continuation_host.service_once()
                assert len(dispatches) == 1
                assert (await continuation_host.aclose()).pending
                assert continuation_host.inspect().failed == 1
                exact = await other.recover_session_continuation(token, context=CONTEXT)
                assert exact.ticket.state == "CONSUMED"
                if advance_before_reconcile:
                    from cayu.sessions import _invocation_lifecycle
                    from cayu.sessions._invocation_lifecycle import (
                        _invocation_lifecycle_receipt_ledger_from_checkpoint,
                    )
                    from cayu.sessions.base import _invocation_lifecycle_authority_read_scope

                    # Two original receipts plus the next admission/release
                    # exactly fill this window. The second successor forces
                    # native compaction without erasing original proof.
                    with monkeypatch.context() as retention_patch:
                        retention_patch.setattr(
                            _invocation_lifecycle,
                            "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS",
                            4,
                        )
                        for turn in range(2):
                            async for _ in other.resume(
                                ResumeRequest(
                                    session_id=session.id,
                                    messages=[
                                        Message.text("user", f"Independent successor {turn}")
                                    ],
                                ),
                                context=CONTEXT,
                            ):
                                pass
                    # Require native compaction even if the injection target moves.
                    with _invocation_lifecycle_authority_read_scope():
                        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(
                            await reopened.load_checkpoint(session.id)
                        )
                    assert len(ledger.receipts) == 4
                    assert len(provider.requests) == 4
                    assert await other.recover_session_continuation(token, context=CONTEXT) == exact
                if disable_before_reconcile:
                    from tests.core.test_participant_lifecycle import change

                    await other.change_participant_lifecycle(
                        change(
                            values[1],
                            participant,
                            key="disable-after-exact-return",
                            revision=1,
                            state="disabled",
                        ),
                        context=CONTEXT,
                    )
            finally:
                unavailable = False
                async with asyncio.timeout(
                    10 if disable_before_reconcile or advance_before_reconcile else 60
                ):
                    while True:
                        try:
                            if not (await continuation_host.aclose()).pending:
                                break
                        except ExceptionGroup as error:
                            failures.append(error)
                        await asyncio.sleep(0.01)
                monkeypatch.setattr(SessionContinuationOwner, "_service_owned", original_service)
            assert failures == [primary]
            assert len(dispatches) == 1
        elif cancel_observer:
            entered = asyncio.Event()
            release = asyncio.Event()
            original_stream = provider.stream

            async def blocked_stream(request):
                entered.set()
                await release.wait()
                async for event in original_stream(request):
                    yield event

            monkeypatch.setattr(provider, "stream", blocked_stream)
            from cayu.collaboration import _host, _host_continuations

            replay_handoffs = []
            acknowledge = _host_continuations.acknowledge_continuation
            readiness_reads = []
            readiness = _host.continuation_delivery_ready

            async def observed_readiness(app, *args, **kwargs):
                if app is other:
                    readiness_reads.append(True)
                return await readiness(app, *args, **kwargs)

            monkeypatch.setattr(_host, "continuation_delivery_ready", observed_readiness)

            def observed_acknowledgement(ownership, outcome):
                result = acknowledge(ownership, outcome)
                if result and outcome.value.admission_commitment is not None:
                    replay_handoffs.append(outcome.value.admission_commitment)
                return result

            monkeypatch.setattr(
                _host_continuations, "acknowledge_continuation", observed_acknowledgement
            )
            running = asyncio.create_task(continuation_host.run())
            competing_host = CollaborationHost(
                application,
                HostRegistration(
                    limits=HostOwnershipLimits(1, 1, 4, 262144),
                    producer_sources=(),
                    producer_rules=(),
                    continuation_rules=(rule,),
                    observation_timeout_s=30,
                    shutdown_timeout_s=30,
                ),
            )
            try:
                await asyncio.wait_for(entered.wait(), 30)
                occupied_reads = len(readiness_reads)
                # Another application/host sees the exact native admission while
                # the first provider is still blocked. Admission is a receiving
                # handoff, not proof that the original invocation has released.
                async with asyncio.timeout(60):
                    while True:
                        observed = await competing_host.service_once()
                        if competing_host._source_errors:
                            raise next(iter(competing_host._source_errors.values()))
                        for outcome in competing_host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        if replay_handoffs:
                            break
                        await asyncio.sleep(0.01)
                assert not (await competing_host.aclose()).pending
                assert continuation_host.inspect().uncertain == 1
                assert len(readiness_reads) == occupied_reads
                assert len(provider.requests) == 1
                # The worker's bounded idle wait may have delivered its own
                # timeout cancellation but not yet exited that timeout scope.
                # Count our new signal separately, then require the internal
                # timeout request to be removed during normal unwinding.
                pending_cancellations = running.cancelling()
                running.cancel()
                assert running.cancelling() == pending_cancellations + 1
                with pytest.raises(asyncio.CancelledError):
                    await running
                assert running.cancelled()
                assert running.cancelling() == 1
                closed = await continuation_host.aclose()
                assert closed.uncertain == 1 and closed.failed == 0
                assert len(provider.requests) == 1
            finally:
                release.set()
                await competing_host.aclose()
                async with asyncio.timeout(60):
                    while (await continuation_host.aclose()).pending:
                        for outcome in continuation_host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        await asyncio.sleep(0.01)
                if not running.done():
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
        else:
            async with continuation_host:
                async with asyncio.timeout(90):
                    while True:
                        observed = await continuation_host.service_once()
                        for outcome in continuation_host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        if observed.serviced:
                            break
                        await asyncio.sleep(0.01)
                # This exact consumed ticket is no longer due. Continuing the
                # host must not repeatedly reserve execution capacity for an
                # admitted replay, even while readonly discovery stays active.
                for _ in range(5):
                    observed = await continuation_host.service_once()
                    assert observed.active == observed.uncertain == 0
                    assert observed.failed == observed.source_failures == 0
                    await asyncio.sleep(0.01)
        assert continuation_host.inspect().uncertain == 0
        assert continuation_host.inspect().failed == 0
        assert len(provider.requests) == 2 + 2 * advance_before_reconcile
        consumed = await other.recover_session_continuation(token, context=CONTEXT)
        assert consumed.ticket.state == "CONSUMED"
        if disable_before_reconcile:
            with pytest.raises(PermissionError):
                async for _ in other.resume(
                    ResumeRequest(session_id=session.id, messages=[Message.text("user", "Next")]),
                    context=CONTEXT,
                ):
                    pass
            assert len(provider.requests) == 2
        if not cancel_observer and not disable_before_reconcile:
            from cayu import ContinuationRecoveryExpectation
            from cayu.runtime._session_continuation_store import digest
            from cayu.sessions.recovery import RecoveryPlanRequest, RecoveryPlanSelection

            consumption = consumed.consumption
            assert consumption is not None
            expected = ContinuationRecoveryExpectation(
                session_instance_id=session.instance_id,
                ticket_key=token.ticket_key,
                record_sha256=digest(consumed.model_dump(mode="json")),
                admission_command_digest=consumption.admission_command_digest,
                admission_expected_run_epoch=consumption.admission_expected_run_epoch,
                profile_digest=consumption.profile_digest,
            )
            selection = RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=(session.id,)),
                continuation=expected,
                participant_context=CONTEXT,
            )
            if advance_before_reconcile:
                # Historical release can discharge the original host observer,
                # but it cannot select a successor invocation for repair.
                with pytest.raises(ValueError, match="no longer owns the admitted invocation"):
                    await other.plan_recovery(selection)
                assert len(provider.requests) == 4
                return
            plan = await other.plan_recovery(selection)
            assert len(plan.items) == 1
            for field, replacement in (
                ("session_instance_id", "different-instance"),
                ("ticket_key", "session-continuation:" + "0" * 64),
                ("record_sha256", "0" * 64),
                ("admission_command_digest", "0" * 64),
                ("admission_expected_run_epoch", consumption.admission_expected_run_epoch + 1),
                ("profile_digest", "0" * 64),
            ):
                with pytest.raises(ValueError):
                    await other.plan_recovery(
                        selection.model_copy(
                            update={
                                "continuation": expected.model_copy(update={field: replacement})
                            }
                        )
                    )
            assert len(provider.requests) == 2
            recovery_host = CollaborationHost(
                other,
                HostRegistration(
                    limits=HostOwnershipLimits(1, 1, 4, 262144),
                    producer_sources=(),
                    producer_rules=(),
                    continuation_rules=(replace(rule, recovery_inactive_for_seconds=1),),
                    observation_timeout_s=30,
                    shutdown_timeout_s=30,
                ),
            )
            async with recovery_host:
                async with asyncio.timeout(60):
                    while True:
                        observed = await recovery_host.service_once()
                        if recovery_host._source_errors:
                            raise next(iter(recovery_host._source_errors.values()))
                        for outcome in recovery_host._owned.inspect().completed:
                            if outcome.error is not None:
                                raise outcome.error
                        # Exact release replay performs no dispatch, so the
                        # pass's serviced counter deliberately remains zero.
                        if recovery_host._continuation_owner._finished:
                            assert observed.serviced == 0
                            break
                        await asyncio.sleep(0.01)
            assert recovery_host.inspect().pending == 0
            assert len(provider.requests) == 2
    finally:
        await other.drain_collaboration_requests()
        await application.drain_collaboration_requests()
        if reopened is not native_stores[1]:
            await reopened.close()
