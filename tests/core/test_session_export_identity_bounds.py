"""Future administrator identity is bounded independently of the original exporter."""

from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_export_content_release import run_async
from tests.core.test_session_exports import OWNER, AcceptanceReader, harness
from tests.core.test_session_exports import backend as backend

from cayu.collaboration._contracts import InitiatorBinding, ObjectRef
from cayu.collaboration._session_export_bounds import MAX_INITIATOR_BYTES
from cayu.collaboration._session_export_store import encoded
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportDenied,
    SessionExportPolicy,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)


class LargeIdentityPolicy(SessionExportPolicy):
    def __init__(self, size):
        self.issuer = OWNER.model_copy(
            update={"owner_id": "\x01" * 512, "incarnation": "\x01" * 512}
        )
        baseline = self.identity("x")
        remaining = size - len(encoded(baseline.model_dump(mode="json")))
        repeats, rest = divmod(remaining, 6)
        self.principal = "x" + "\x01" * repeats + "a" * rest
        assert len(self.principal) <= 512
        assert len(encoded(self.identity(self.principal).model_dump(mode="json"))) == size

    def identity(self, principal):
        return InitiatorBinding(
            issuer=self.issuer,
            principal=principal,
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        )

    @property
    def ref(self):
        return ObjectRef(
            owner=self.issuer, kind="policy", object_id="large", incarnation="v1", revision=1
        )

    @asynccontextmanager
    async def acquire(self, context, **kwargs):
        if context.principal != self.principal:
            raise SessionExportDenied()
        yield SessionExportAuthorization(
            issuer=self.issuer,
            principal=self.principal,
            policy=self.ref,
            revision=1,
            expires_at_ms=4102444800000,
        )


@pytest.mark.parametrize("offset", [-1, 0, 1])
@run_async
async def test_public_authority_identity_boundary_includes_json_escaping(backend, offset):
    async with harness(backend) as case:
        policy = LargeIdentityPolicy(MAX_INITIATOR_BYTES + offset)
        app, store, _, _ = case.app(policy=policy)
        await case.create(store)
        context = SessionExportAccessContext(principal=policy.principal)
        if offset > 0:
            with pytest.raises(SessionExportUnavailable):
                await app.initialize_session_exports(case.session_id, context=context)
        else:
            namespace = await app.initialize_session_exports(case.session_id, context=context)
            assert namespace.session_id == case.session_id


@run_async
async def test_small_export_reserves_large_independent_administrator_settlement(backend):
    from tests.core.test_session_exports import CONTEXT

    async with harness(backend) as case:
        app, store, _, _ = case.app()
        await case.create(store)
        request = await case.request(app)
        exported = await app.export_session(request, context=CONTEXT)
        policy = LargeIdentityPolicy(MAX_INITIATOR_BYTES)
        administrator, _, _, _ = case.app(policy=policy, readers=(AcceptanceReader(),))
        context = SessionExportAccessContext(principal=policy.principal)
        settlement = SessionExportSettlementRequest(
            request=request,
            mode="release",
            operation=request.ref.operation.model_copy(update={"caller_key": "\x01" * 512}),
        )
        receipt = await administrator.settle_session_export(settlement, context=context)
        assert receipt.acceptance.export_receipt == exported
        assert len(encoded(receipt.initiator.model_dump(mode="json"))) == MAX_INITIATOR_BYTES
        assert await administrator.settle_session_export(settlement, context=context) == receipt
        await store.delete_session(case.session_id)
