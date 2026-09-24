"""Public administrative readback after native append acknowledgement loss."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable


async def recover_delivery(application, *, expected_status):
    with pytest.raises(CollaborationAccessDenied):
        await application.list_pending_clarification_deliveries(
            context=CONTEXT.model_copy(update={"principal": "outsider"})
        )
    page = await application.list_pending_clarification_deliveries(context=CONTEXT, limit=1)
    assert len(page.items) == 1
    for invalid in (0, 65, True):
        with pytest.raises(CollaborationContractError):
            await application.list_pending_clarification_deliveries(context=CONTEXT, limit=invalid)
    assert page.next_cursor is not None
    assert not (
        await application.list_pending_clarification_deliveries(
            context=CONTEXT, limit=1, cursor=page.next_cursor
        )
    ).items
    serialized = page.model_dump_json()
    for private in ("Which API version?", "private-state", "append_key", "permit", "payload"):
        assert private not in serialized
    selected = type(page.items[0].recovery).model_validate_json(
        page.items[0].recovery.model_dump_json()
    )
    with pytest.raises(CollaborationConflict):
        await application.reconcile_clarification_delivery(
            selected.model_copy(update={"intent_sha256": "f" * 64}), context=CONTEXT
        )
    with pytest.raises(CollaborationUnavailable):
        await application.reconcile_clarification_delivery(
            selected.model_copy(
                update={
                    "operation": selected.operation.model_copy(
                        update={"caller_key": "absent-delivery"}
                    )
                }
            ),
            context=CONTEXT,
        )
    with pytest.raises(CollaborationAccessDenied):
        await application.reconcile_clarification_delivery(
            selected, context=CONTEXT.model_copy(update={"principal": "outsider"})
        )
    assert await application.list_pending_clarification_deliveries(context=CONTEXT, limit=1) == page
    result = await application.reconcile_clarification_delivery(selected, context=CONTEXT)
    assert result.status == expected_status
    assert await application.reconcile_clarification_delivery(selected, context=CONTEXT) == result
    remaining = await application.list_pending_clarification_deliveries(context=CONTEXT)
    assert bool(remaining.items) == (expected_status == "pending")
    return result


async def cancel_delivery_recovery(application, *, request, monkeypatch):
    page = await application.list_pending_clarification_deliveries(context=CONTEXT)
    assert len(page.items) == 1
    selector = page.items[0].recovery
    store = application.session_store
    original = type(store).read_peer_content_attempt
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(self, candidate):
        result = await original(self, candidate)
        if candidate.operation_key == request.operation_key:
            assert result is not None and result.status == "appended"
            entered.set()
            await release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(type(store), "read_peer_content_attempt", held)
        observer = asyncio.create_task(
            application.reconcile_clarification_delivery(selector, context=CONTEXT)
        )
        ready = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait((observer, ready), return_when=asyncio.FIRST_COMPLETED), 30
            )
            if not entered.is_set():
                await observer
                pytest.fail("Recovery did not reach exact native append readback.")
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
            assert await application.list_pending_clarification_deliveries(context=CONTEXT) == page
            release.set()
            result = await application.reconcile_clarification_delivery(selector, context=CONTEXT)
            assert result.status == "appended"
            assert (
                await application.reconcile_clarification_delivery(selector, context=CONTEXT)
                == result
            )
            assert not (
                await application.list_pending_clarification_deliveries(context=CONTEXT)
            ).items
        finally:
            release.set()
            ready.cancel()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, ready, return_exceptions=True)
