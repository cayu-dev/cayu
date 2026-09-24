"""Public two-instance lifecycle ordering at the existing durable permit owner."""

import asyncio
from contextlib import suppress

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_lifecycle import change

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority


async def disable_service(
    app,
    other,
    initialized,
    request,
    *,
    context,
    delivery_context,
    registered_first,
    payloads,
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    registrations = []
    count = len(payloads)
    target_id = request.delivery.append.append_key.target_session_id
    before = await other.session_store.load(target_id)
    original = TemporaryServicePermitAuthority.register

    async def held(authority, candidate):
        if candidate.dispatch.intent.operation != request.operation:
            return await original(authority, candidate)
        if registered_first:
            registered = await original(authority, candidate)
            registrations.append(registered)
        entered.set()
        await release.wait()
        if registered_first:
            return registered
        return await original(authority, candidate)

    with monkeypatch.context() as patch:
        patch.setattr(TemporaryServicePermitAuthority, "register", held)
        observer = asyncio.create_task(
            app.service_clarification(request, context=context, delivery_context=delivery_context)
        )
        ready = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait((observer, ready), return_when=asyncio.FIRST_COMPLETED), 90
            )
            if not entered.is_set():
                await observer
                pytest.fail("Public service did not reach the registration boundary.")
            prepared = await other.session_store.load(target_id)
            # Native preparation persists an inert reconciliation record and
            # touches storage timestamps. It must not change execution state.
            timestamps = {"last_activity_at", "updated_at"}
            assert prepared.model_dump(exclude=timestamps) == before.model_dump(exclude=timestamps)
            participant = request.delivery.recipient
            inspected = await other.inspect_participant(participant, context=CONTEXT)
            await other.change_participant_lifecycle(
                change(
                    initialized,
                    participant,
                    key="disable-service-at-permit-boundary",
                    revision=inspected.participant.lifecycle_revision,
                    state="disabled",
                ),
                context=CONTEXT,
            )
            assert len(payloads) == count
            assert await other.session_store.load(target_id) == prepared
            release.set()
            with suppress(CollaborationConflict, CollaborationUnavailable, PermissionError):
                await asyncio.wait_for(observer, 120)
            # Administrative settlement does not require renewed execution or
            # disclosure authority. Do not retry public service with a new key.
            async with asyncio.timeout(120):
                while True:
                    try:
                        page = await other.inspect_clarification_services(
                            request.ticket, context=CONTEXT
                        )
                    except CollaborationUnavailable:
                        # Inspection deliberately refuses a mixed index/child
                        # snapshot while the original owner publishes its return.
                        await asyncio.sleep(0.05)
                        continue
                    assert len(page.items) == 1
                    selected = page.items[0]
                    if not registered_first:
                        assert selected.state == "prepared"
                        result = await other.exclude_clarification_service(
                            selected.recovery, context=CONTEXT
                        )
                        assert result.state == "excluded"
                        excluded = await other.session_store.load(target_id)
                        assert excluded.model_dump(exclude=timestamps) == before.model_dump(
                            exclude=timestamps
                        )
                        break
                    try:
                        result = await other.reconcile_clarification_service(
                            selected.recovery, context=CONTEXT
                        )
                    except CollaborationUnavailable:
                        # Observation expiry does not finish the retained owner.
                        # Retry the same recovery selection, never service again.
                        await asyncio.sleep(0.05)
                        continue
                    if result.state == "returned":
                        assert registrations
                        assert result.released_session_status == "failed"
                        # Admission advances once; its proven RELEASE advances
                        # the writer epoch once more. Neither is a second turn.
                        assert (await other.session_store.load(target_id)).run_epoch == (
                            before.run_epoch + 2
                        )
                        break
                    await asyncio.sleep(0.05)
            assert len(payloads) == count
            assert (
                await other.reconcile_clarification_service(selected.recovery, context=CONTEXT)
                == result
            )
            assert len(payloads) == count
        finally:
            release.set()
            ready.cancel()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, ready, return_exceptions=True)
