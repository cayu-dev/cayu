"""A dispatched ordinary invocation retains its writer against clarification."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration.participants import CollaborationUnavailable
from cayu.messages import Message
from cayu.sessions.requests import ResumeRequest


async def reject_busy_target(
    app, request, *, context, delivery_context, payloads, entered, release
):
    target = request.delivery.append.append_key.target_session_id
    count = len(payloads)

    async def execute():
        return [
            event
            async for event in app.resume(
                ResumeRequest(session_id=target, messages=[Message.text("user", "Continue.")]),
                context=CONTEXT,
            )
        ]

    worker = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(entered.wait(), 60)
        before = await app.session_store.load(target)
        events = await app.session_store.load_events(target)
        assert before is not None and before.status.value == "running"
        with pytest.raises(CollaborationUnavailable):
            await app.service_clarification(
                request, context=context, delivery_context=delivery_context
            )
        after = await app.session_store.load(target)
        assert after is not None and after.run_epoch == before.run_epoch
        assert after.status == before.status
        assert await app.session_store.load_events(target) == events
        assert len(payloads) == count + 1
        page = await app.inspect_clarification_services(request.ticket, context=CONTEXT)
        for item in page.items:
            assert item.state in {"prepared", "excluded"}
            await app.exclude_clarification_service(item.recovery, context=CONTEXT)
        retained = await app.session_store.load_continuation_ticket(
            request.ticket.session_id,
            session_instance_id=request.ticket.session_instance_id,
            registration_key=request.ticket.registration_key,
        )
        assert retained.ticket.state == "WAITING"
        assert retained.latch is None and retained.consumption is None
        release.set()
        await asyncio.wait_for(worker, 60)
        assert len(payloads) == count + 1
    finally:
        release.set()
        await asyncio.gather(worker, return_exceptions=True)
