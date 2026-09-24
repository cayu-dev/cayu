"""Real final-result election competes with retained public temporary service."""

import asyncio
from contextlib import suppress

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.events import EventType
from cayu.runtime._session_continuation import ContinuationConflict
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.sessions.base import CompactSessionRequest


async def arbitrate_latch(
    app,
    request,
    *,
    timing,
    context,
    delivery_context,
    publish,
    consume,
    payloads,
    monkeypatch,
):
    count = len(payloads)
    ticket = request.ticket

    async def read_ticket():
        return await app.session_store.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )

    async def reconcile_return(recovery):
        async with asyncio.timeout(120):
            while True:
                try:
                    result = await app.reconcile_clarification_service(recovery, context=CONTEXT)
                except CollaborationUnavailable:
                    # Observation expiry does not exclude the owned settlement.
                    # Keep the exact selector until positive native return.
                    await asyncio.sleep(0.25)
                    continue
                if result.state == "returned":
                    return result
                assert result.state != "excluded"
                await asyncio.sleep(0.25)

    if timing == "foreign_settlement":
        from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority

        entered = asyncio.Event()
        release = asyncio.Event()
        original = TemporaryServicePermitAuthority.settle

        async def fail_settlement(owner, candidate, *, reader):
            if candidate.dispatch.intent.operation != request.operation:
                return await original(owner, candidate, reader=reader)
            entered.set()
            await release.wait()
            raise ConnectionError("Foreign settlement unavailable after native return")

        with monkeypatch.context() as patch:
            patch.setattr(TemporaryServicePermitAuthority, "settle", fail_settlement)
            observer = asyncio.create_task(
                app.service_clarification(
                    request, context=context, delivery_context=delivery_context
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 120)
                returned = await read_ticket()
                assert returned.ticket.state == "WAITING"
                latched = await publish()
                await consume(latched, blocked=False)
                assert (await read_ticket()).ticket.state == "CONSUMED"
                with pytest.raises(ContinuationConflict, match="settlement acknowledgement"):
                    await app.validate_session_closure(ticket.session_id)
                with pytest.raises(ContinuationConflict, match="settlement acknowledgement"):
                    await app.session_store.delete_session(ticket.session_id)
            finally:
                release.set()
                with suppress(CollaborationConflict, CollaborationUnavailable):
                    await asyncio.wait_for(observer, 120)
        page = await app.inspect_clarification_services(ticket, context=CONTEXT)
        assert len(page.items) == 1
        result = await reconcile_return(page.items[0].recovery)
        assert result.state == "returned"
        assert (
            await app.reconcile_clarification_service(page.items[0].recovery, context=CONTEXT)
            == result
        )
        await app.validate_session_closure(ticket.session_id)
        assert len(payloads) == count + 2
        return

    if timing == "before":
        latched = await publish()
        assert latched.ticket.state == "WAITING"
        with suppress(CollaborationConflict, CollaborationUnavailable):
            result = await app.service_clarification(
                request, context=context, delivery_context=delivery_context
            )
            assert result.state in {"prepared", "excluded"}
        assert len(payloads) == count
        retained = await read_ticket()
        assert retained.latch == latched.latch and retained.consumption is None
        page = await app.inspect_clarification_services(ticket, context=CONTEXT)
        for item in page.items:
            assert item.state in {"prepared", "excluded"}
            await app.exclude_clarification_service(item.recovery, context=CONTEXT)
        await consume(await read_ticket(), blocked=False)
        assert len(payloads) == count + 1
        return

    entered = asyncio.Event()
    release = asyncio.Event()
    failed = asyncio.Event()
    reconcile = SessionContinuationOwner.reconcile_temporary

    async def held(owner, candidate):
        if candidate.dispatch.intent.operation != request.operation:
            return await reconcile(owner, candidate)
        if failed.is_set():
            return await reconcile(owner, candidate)
        # The provider has genuinely released its invocation, but the source
        # ticket still owns admitted service responsibility. Native release
        # alone does not consume the wait or settle the return handoff.
        async with asyncio.timeout(120):
            while True:
                native = await owner.store._read_temporary_continuation_outcome(candidate)
                if native is not None and native.state == "returned":
                    break
                await asyncio.sleep(0.05)
        entered.set()
        await release.wait()
        failed.set()
        raise ConnectionError("return observation failed after final latch publication")

    with monkeypatch.context() as patch:
        patch.setattr(SessionContinuationOwner, "reconcile_temporary", held)
        observer = asyncio.create_task(
            app.service_clarification(request, context=context, delivery_context=delivery_context)
        )
        try:
            await asyncio.wait_for(entered.wait(), 120)
            assert len(payloads) == count + 1
            before = await read_ticket()
            assert before.ticket.state == "SERVICING" and before.latch is None
            # Native RELEASE has happened, so a terminal session status alone
            # would permit closure. The still-unacknowledged source/receiving
            # responsibility must reject the public cleanup entrance instead.
            for session_id in {
                ticket.session_id,
                request.delivery.append.append_key.target_session_id,
            }:
                snapshot = await app.session_store.load(session_id)
                events = await app.session_store.load_events(session_id)
                with pytest.raises(ContinuationConflict):
                    await app.validate_session_closure(session_id)
                assert await app.session_store.load(session_id) == snapshot
                assert await app.session_store.load_events(session_id) == events
            assert await read_ticket() == before
            # The receiving session has its initial turn plus the actual service
            # turn. A side-session's parked parent still has only one turn and
            # therefore has no older complete context eligible for compaction.
            target_id = request.delivery.append.append_key.target_session_id
            receiving = await app.session_store.load(target_id)
            transcript = await app.session_store.load_transcript_snapshot(target_id)
            assert receiving is not None
            compacted = [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=receiving.id,
                        idempotency_key="pending-service-compaction",
                        expected_run_epoch=receiving.run_epoch,
                        expected_transcript_cursor=transcript.cursor,
                    ),
                    context=CONTEXT,
                )
            ]
            assert any(event.type is EventType.SESSION_CHECKPOINTED for event in compacted)
            assert await read_ticket() == before
            assert await app.session_store.load_transcript_snapshot(receiving.id) == transcript
            assert len(payloads) == count + 1
            latched = await publish()
            assert latched.ticket.state == "SERVICING"
            await consume(latched, blocked=True)
            release.set()
            await asyncio.wait_for(failed.wait(), 120)
            with suppress(CollaborationConflict, CollaborationUnavailable):
                await asyncio.wait_for(observer, 120)
            after_failure = await read_ticket()
            assert after_failure.latch == latched.latch
            assert after_failure.ticket.state == "SERVICING"
            assert after_failure.consumption is None
            page = await app.inspect_clarification_services(ticket, context=CONTEXT)
            assert len(page.items) == 1 and page.items[0].state == "admitted"
            result = await reconcile_return(page.items[0].recovery)
            assert result.state == "returned"
            assert result.released_session_status == "completed"
            assert (
                await app.reconcile_clarification_service(page.items[0].recovery, context=CONTEXT)
                == result
            )
            assert len(payloads) == count + 1
            returned = await read_ticket()
            assert returned.ticket.state == "WAITING" and returned.latch == latched.latch
            await consume(returned, blocked=False)
            assert len(payloads) == count + 2
        finally:
            release.set()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
