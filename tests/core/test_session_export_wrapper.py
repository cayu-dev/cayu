"""New export routes traverse the checkpoint wrapper, not only capability probes."""

import pytest
from tests.core.test_session_export_content_release import ReviewOwner, reviewed_request, run_async
from tests.core.test_session_export_mandates import Resolver
from tests.core.test_session_exports import CONTEXT, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration._session_export_coordinator import SessionExportCoordinator
from cayu.collaboration.exports import SessionExportSettlementRequest, SessionExportUnavailable
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store


@pytest.mark.parametrize("mode", ["mandate", "reviewed_prose", "combined"])
@run_async
async def test_extended_export_routes_through_checkpoint_wrapper(backend, mode):
    def wrap_owner(app):
        # The checkpoint proxy is an internal store boundary, not a public
        # SessionStore subtype. Exercise the real coordinator through this
        # supported wrapper without changing the public constructor contract.
        original = app._session_export_coordinator
        app._session_export_coordinator = SessionExportCoordinator(
            store=runtime_checkpoint_session_store(app.session_store),
            registration=original.registration,
            redactor=original.redactor,
        )

    async with harness(backend) as case:
        plain, store, _, _ = case.app()
        wrap_owner(plain)
        await case.create(store)
        request = await case.request(plain)
        reader = None
        if mode != "mandate":
            reader = ReviewOwner()
            request, _ = await reviewed_request(case, plain, store, reader)
        resolver = Resolver(request) if mode != "reviewed_prose" else None
        context = resolver.context if resolver is not None else CONTEXT
        app, _, _, projector = case.app(
            mandates=resolver, release_readers=() if reader is None else (reader,)
        )
        wrap_owner(app)
        receipt = await app.export_session(request, context=context)
        assert await app.read_session_export(request, context=context) == (
            {"count": 5} if reader is None else {"text": "Reviewed advice."}
        )
        assert projector.calls == (1 if reader is None else 0)
        reopened, _, _, _ = case.app(mandates=resolver, projectors=())
        wrap_owner(reopened)
        assert await reopened.export_session(request, context=context) == receipt
        assert (await reopened.lookup_session_export(request, context=context)).receipt == receipt
        settlement = SessionExportSettlementRequest(
            request=request,
            operation=request.ref.operation.model_copy(update={"caller_key": "retire"}),
            mode="retire",
        )
        settled = await reopened.settle_session_export(settlement, context=context)
        assert await reopened.settle_session_export(settlement, context=context) == settled
        with pytest.raises(SessionExportUnavailable):
            await reopened.read_session_export(request, context=context)
        assert len(await published(store, case.session_id)) == 1
        await store.delete_session(case.session_id)
