"""Revoked-source cleanup retains ownership through cancellation and lost ACK."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT


async def exclude_with_lost_ack(application, selector, peer, monkeypatch):
    store = application.session_store
    exclude = store.exclude_peer_content
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_exclusion(*args, **kwargs):
        await exclude(*args, **kwargs)
        entered.set()
        await release.wait()
        raise ConnectionError("exclusion committed before acknowledgement loss")

    async def delayed_append():
        # Simulate the receiving write of an already-dispatched append arriving
        # after exclusion. Native fencing must reject it even if its sender had
        # passed disclosure authorization before revocation.
        await release.wait()
        return await store.append_peer_content(peer, qualify_target=lambda session: None)

    with monkeypatch.context() as patch:
        patch.setattr(store, "exclude_peer_content", held_exclusion)
        late = asyncio.create_task(delayed_append())
        observer = asyncio.create_task(
            application.exclude_clarification_delivery(selector, context=CONTEXT)
        )
        ready = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait((ready, observer), return_when=asyncio.FIRST_COMPLETED), 30
            )
            if not entered.is_set():
                await observer
                pytest.fail("Cleanup did not reach native exclusion.")
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
            assert (await application.list_pending_clarification_deliveries(context=CONTEXT)).items
            release.set()
            assert (await asyncio.wait_for(late, 30)).status == "excluded"
        finally:
            release.set()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(ready, observer, late, return_exceptions=True)
