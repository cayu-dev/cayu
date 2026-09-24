"""A produced public reply cannot overwrite a competing question cancellation."""

import pytest

from cayu.collaboration._contracts import CollaborationConflict


async def arbitrate_reply_cancel(
    app, opening, reply, close, *, cancel_first, context, export_context, ticket, payloads
):
    count = len(payloads)
    before = await app.inspect_collaboration_request(reply.expected, context=context)

    async def accept():
        return await app.reply_to_clarification(reply, context=export_context)

    async def cancel():
        return await app.close_clarification(close, context=context)

    winner, loser = (cancel, accept) if cancel_first else (accept, cancel)
    result = await winner()
    stable = await app.inspect_collaboration_request(reply.expected, context=context)
    with pytest.raises(CollaborationConflict):
        await loser()
    assert await winner() == result
    assert await app.inspect_collaboration_request(reply.expected, context=context) == stable
    assert stable.state == before.state == "open"
    assert stable.clarification.input_revision == before.clarification.input_revision + (
        not cancel_first
    )
    question = await app.inspect_clarification(opening, context=context)
    assert question.status == "match"
    assert question.receipt.state == ("cancelled" if cancel_first else "answered")
    retained = await app.session_store.load_continuation_ticket(
        ticket.session_id,
        session_instance_id=ticket.session_instance_id,
        registration_key=ticket.registration_key,
    )
    assert retained.ticket.state == "WAITING"
    assert retained.latch is None and retained.consumption is None
    assert len(payloads) == count
