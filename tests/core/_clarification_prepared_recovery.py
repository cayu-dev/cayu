"""Two public applications fence delayed real runtime preparation."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority


async def recover_prepared(
    app, other, request, *, context, delivery_context, phase, payloads, monkeypatch
):
    entered = asyncio.Event()
    release = asyncio.Event()
    count = len(payloads)
    with monkeypatch.context() as patch:
        if (
            phase == "after_preparation"
            and request.delivery.append.append_key.target_session_id != request.ticket.session_id
        ):
            original_pair = type(app.session_store)._prepare_temporary_side_service

            async def held_pair(self, preparation):
                result = await original_pair(self, preparation)
                if preparation.dispatch.intent.operation == request.operation:
                    entered.set()
                    await release.wait()
                return result

            patch.setattr(type(app.session_store), "_prepare_temporary_side_service", held_pair)
        elif phase == "after_preparation":
            original_publish = type(app.session_store)._publish_temporary_continuation_service

            async def held(self, *, previous, proposed):
                result = await original_publish(self, previous=previous, proposed=proposed)
                if (
                    previous is None
                    and proposed.state == "prepared"
                    and proposed.intent.operation == request.operation
                ):
                    entered.set()
                    await release.wait()
                return result

            patch.setattr(type(app.session_store), "_publish_temporary_continuation_service", held)
        else:
            original_register = TemporaryServicePermitAuthority.register

            async def held(self, candidate):
                if candidate.dispatch.intent.operation == request.operation:
                    entered.set()
                    await release.wait()
                return await original_register(self, candidate)

            patch.setattr(TemporaryServicePermitAuthority, "register", held)
        observer = asyncio.create_task(
            app.service_clarification(request, context=context, delivery_context=delivery_context)
        )
        ready = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait((observer, ready), return_when=asyncio.FIRST_COMPLETED), 60
            )
            if not entered.is_set():
                await observer
                pytest.fail("Public service did not reach native preparation.")
            assert len(payloads) == count
            assert not (await other.list_pending_clarification_services(context=CONTEXT)).items
            with pytest.raises(CollaborationAccessDenied):
                await other.inspect_clarification_services(
                    request.ticket, context=CONTEXT.model_copy(update={"principal": "outsider"})
                )
            page = await other.inspect_clarification_services(request.ticket, context=CONTEXT)
            assert len(page.items) == 1 and page.items[0].state == "prepared"
            selector = type(page.items[0].recovery).model_validate_json(
                page.items[0].recovery.model_dump_json()
            )
            assert (
                await other.reconcile_clarification_service(selector, context=CONTEXT)
            ).state == "prepared"
            with pytest.raises(CollaborationConflict):
                await other.exclude_clarification_service(
                    selector.model_copy(update={"dispatch_sha256": "e" * 64}), context=CONTEXT
                )
            async with asyncio.timeout(120):
                while True:
                    try:
                        excluded = await other.exclude_clarification_service(
                            selector, context=CONTEXT
                        )
                        break
                    except CollaborationUnavailable:
                        # The exact owned exclusion can outlive its observation
                        # window. Do not release the delayed admission until its
                        # positive exclusion receipt has been reconstructed.
                        await asyncio.sleep(0.05)
            assert excluded.state == "excluded" and excluded.released_session_status is None
            assert await other.exclude_clarification_service(selector, context=CONTEXT) == excluded
            assert len(payloads) == count
            release.set()
            try:
                result = await asyncio.wait_for(observer, 60)
            except (CollaborationConflict, CollaborationUnavailable):
                pass
            else:
                assert result == excluded
            assert len(payloads) == count
            assert (
                await other.reconcile_clarification_service(selector, context=CONTEXT) == excluded
            )
            inspected = await other.inspect_clarification_services(request.ticket, context=CONTEXT)
            assert len(inspected.items) == 1 and inspected.items[0].state == "excluded"
        finally:
            release.set()
            ready.cancel()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, ready, return_exceptions=True)
