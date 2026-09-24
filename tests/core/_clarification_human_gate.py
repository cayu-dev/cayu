"""A genuine human-input pause is not a clarification-service permit."""

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store


async def reject_human_paused(app, request, *, context, delivery_context, payloads):
    target = request.delivery.append.append_key.target_session_id
    store = runtime_checkpoint_session_store(app.session_store)
    checkpoint = await store.load_checkpoint(target)
    assert checkpoint is not None and checkpoint.get("pending_user_input") is not None
    pending = checkpoint["pending_user_input"]
    before = await app.session_store.load(target)
    count = len(payloads)
    with pytest.raises(CollaborationUnavailable):
        await app.service_clarification(request, context=context, delivery_context=delivery_context)
    after = await app.session_store.load(target)
    assert after is not None and before is not None
    assert after.run_epoch == before.run_epoch and after.status == before.status
    current = await store.load_checkpoint(target)
    assert current is not None and current["pending_user_input"] == pending
    assert len(payloads) == count
    page = await app.inspect_clarification_services(request.ticket, context=CONTEXT)
    for item in page.items:
        assert item.state in {"prepared", "excluded"}
        await app.exclude_clarification_service(item.recovery, context=CONTEXT)
    returned = await app.session_store.load_continuation_ticket(
        request.ticket.session_id,
        session_instance_id=request.ticket.session_instance_id,
        registration_key=request.ticket.registration_key,
    )
    assert returned.ticket.state == "WAITING"
    assert returned.latch is None and returned.consumption is None
    current = await store.load_checkpoint(target)
    assert current is not None and current["pending_user_input"] == pending
    assert len(payloads) == count
