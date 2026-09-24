"""Public terminal-wait cleanup after real service, with reconstructed owners."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._session_continuation import ContinuationConflict, ContinuationUnavailable
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


async def retire_terminal_wait(
    app,
    service_request,
    *,
    mode,
    service_context,
    delivery_context,
    wait,
    wait_context,
    creation,
    participant,
    application_for,
    collaboration_factory,
    session_factory,
    backend,
    payloads,
    monkeypatch,
):
    ticket = service_request.ticket
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": ticket.session_id}),
        session_instance_id=ticket.session_instance_id,
        execution_key="park-question-target",
    )
    count = len(payloads)
    entered, release = asyncio.Event(), asyncio.Event()
    returned = asyncio.Event()
    return_errors = []
    settlement_entered, settlement_release = asyncio.Event(), asyncio.Event()
    reconcile = SessionContinuationOwner.reconcile_temporary
    settle = TemporaryServicePermitAuthority.settle

    async def held_settlement(owner, candidate, *, reader):
        if candidate.dispatch.intent.operation == service_request.operation:
            settlement_entered.set()
            await settlement_release.wait()
        return await settle(owner, candidate, reader=reader)

    async def held(owner, candidate):
        if candidate.dispatch.intent.operation == service_request.operation:
            async with asyncio.timeout(360):
                while True:
                    outcome = await owner.store._read_temporary_continuation_outcome(candidate)
                    if outcome is not None and outcome.state == "returned":
                        break
                    await asyncio.sleep(0.05)
            entered.set()
            await release.wait()
        try:
            # The receiver's own bounded observation can also expire while its
            # ACK write remains owned. Wait for exact settlement, not merely
            # for the wrapper to return an observation-timeout exception.
            async with asyncio.timeout(360):
                while True:
                    try:
                        return await reconcile(owner, candidate)
                    except ContinuationUnavailable:
                        await asyncio.sleep(0.05)
        except BaseException as error:
            return_errors.append(error)
            raise
        finally:
            returned.set()

    async def read(application):
        result = await application.session_store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        assert result is not None
        return result

    async def cleanup(application, candidate=execution, context=CONTEXT):
        return await application.exclude_participant_session_wait(
            candidate,
            wait,
            participant=participant,
            context=context,
            wait_context=wait_context,
        )

    with monkeypatch.context() as patch:
        patch.setattr(SessionContinuationOwner, "reconcile_temporary", held)
        patch.setattr(TemporaryServicePermitAuthority, "settle", held_settlement)
        observer = asyncio.create_task(
            app.service_clarification(
                service_request,
                context=service_context,
                delivery_context=delivery_context,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 360)
            before = await read(app)
            assert before.ticket.state == "SERVICING"
            bound = wait.model_copy(update={"delivery_ticket": before.preparation.intent})
            source = app._participant_coordinator._ready()[0]
            if mode == "expired":
                # Advance the authoritative store clock across the immutable
                # deadline, not a worker timer or manually constructed receipt.
                transaction = source._transaction
                now = int(datetime.fromisoformat(wait.deadline).timestamp() * 1000) + 1

                @asynccontextmanager
                async def expired_clock(*args, **kwargs):
                    async with transaction(*args, **kwargs) as tx:

                        async def owner_now():
                            return now

                        with monkeypatch.context() as clock_patch:
                            clock_patch.setattr(tx, "now_ms", owner_now)
                            yield tx

                with monkeypatch.context() as clock_patch:
                    clock_patch.setattr(source, "_transaction", expired_clock)
                    terminal = await app.cancel_collaboration_wait(
                        bound,
                        context=wait_context,
                        expired=True,
                    )
            else:
                terminal = await app.cancel_collaboration_wait(bound, context=wait_context)
            assert terminal.state == mode and terminal.delivery == "pending"
            with pytest.raises(ContinuationConflict):
                await cleanup(app)
            assert await read(app) == before
            release.set()
            await asyncio.wait_for(settlement_entered.wait(), 360)
            native_return = await read(app)
            assert native_return.ticket.state == "WAITING"
            # Native return alone is not foreign responsibility settlement.
            with pytest.raises(ContinuationConflict):
                await cleanup(app)
            assert await read(app) == native_return
        finally:
            release.set()
            settlement_release.set()
            with suppress(CollaborationUnavailable, ContinuationUnavailable):
                await asyncio.wait_for(observer, 360)
            # Public observation can time out while the owned receiving write
            # continues. Wait for the actual return/ACK, not that observation.
            await asyncio.wait_for(returned.wait(), 360)
            if return_errors:
                raise return_errors[0]
    page = await app.inspect_clarification_services(ticket, context=CONTEXT)
    assert len(page.items) == 1
    result = await app.reconcile_clarification_service(page.items[0].recovery, context=CONTEXT)
    assert result.state == "returned"
    assert len(payloads) == count + 1

    other_store = app.session_store if backend == "memory" else session_factory()
    other_source = collaboration_factory()
    other = application_for(other_source, other_store)
    await other.initialize_collaboration()
    try:
        before = await read(other)
        assert before.ticket.state == "WAITING"
        with pytest.raises(PermissionError):
            await cleanup(other, context=CONTEXT.model_copy(update={"principal": "outsider"}))
        with pytest.raises(ContinuationConflict):
            await cleanup(other, replace(execution, execution_key="wrong"))
        assert await read(other) == before
        retire = SessionContinuationOwner.retire_released

        async def lost_native_ack(owner, candidate):
            await retire(owner, candidate)
            raise OSError("native retirement committed before acknowledgement loss")

        with monkeypatch.context() as patch:
            patch.setattr(SessionContinuationOwner, "retire_released", lost_native_ack)
            with pytest.raises(OSError):
                await cleanup(other)
        retired = await read(other)
        assert retired.ticket.state == "RETIRED" and not retired.retirement_acknowledged
        assert retired.retirement.reason == mode
        with pytest.raises(ContinuationConflict):
            await other.validate_session_closure(ticket.session_id)
        # Lose the observing application as well as its acknowledgement. The
        # next owner has no runtime invocation or provider registration.
        if backend != "memory":
            await other._request_coordinator.close()
            await other_store.close()
            await other_source.close()
            other_store = session_factory()
        other_source = collaboration_factory()
        other = application_for(other_source, other_store)
        await other.initialize_collaboration()
        assert await read(other) == retired
        record_delivery = other_source.record_wait_delivery

        async def lost_foreign_ack(*args, **kwargs):
            await record_delivery(*args, **kwargs)
            raise OSError("foreign exclusion committed before acknowledgement loss")

        with monkeypatch.context() as patch:
            patch.setattr(other_source, "record_wait_delivery", lost_foreign_ack)
            with pytest.raises(OSError):
                await cleanup(other)
        assert await read(other) == retired
        receipt = await cleanup(other)
        assert await cleanup(other) == receipt
        settled = await read(other)
        assert settled.retirement == retired.retirement and settled.retirement_acknowledged
        assert (
            await other.inspect_collaboration_wait(bound, context=wait_context)
        ).delivery == "excluded"
        assert len(payloads) == count + 1
    finally:
        await other._request_coordinator.close()
        if backend != "memory":
            await other_store.close()
            await other_source.close()
