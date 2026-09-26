"""Retired source history is necessary, but not sufficient, maintenance authority."""

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_reclamation_ownership import retired_producer

from cayu.collaboration.access import CollaborationAccessDenied


@pytest.mark.anyio
async def test_public_reclamation_requires_current_scope_and_bounded_batch(
    native_stores, monkeypatch
):
    app, provider, namespace = await retired_producer(native_stores, monkeypatch)
    receiver = app._request_coordinator._registration.receiving_owner
    original = receiver._retire_producer_cleanup
    calls = []

    async def observed(retirement, *, authority, limit):
        calls.append((retirement, limit))
        return await original(retirement, authority=authority, limit=limit)

    monkeypatch.setattr(receiver, "_retire_producer_cleanup", observed)
    policy = app._participant_coordinator._registration.access_policy
    try:
        for limit in (True, False, 0, -1, 33, 2**53, 1.0):
            with pytest.raises(ValueError, match="batch"):
                await app.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=limit)
        policy.denied.add("namespace_prune")
        with pytest.raises(CollaborationAccessDenied):
            await app.reclaim_producer_cleanup(namespace, context=CONTEXT)
        policy.denied.clear()
        # A grant for no participants cannot become a whole-namespace grant.
        policy.allowed = ()
        with pytest.raises(CollaborationAccessDenied):
            await app.reclaim_producer_cleanup(namespace, context=CONTEXT)
        policy.allowed = None
        assert calls == []
        result = await app.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=1)
        assert result.removed == 1 and not result.remaining
        assert len(calls) == 1 and calls[0][1] == 1
        # Exact serialized retirement data does not carry runtime provenance.
        with pytest.raises(PermissionError):
            await native_stores[1]._retire_native_producer_cleanup(
                result.retirement.model_copy(deep=True), authority=None, limit=1
            )
        replay = await app.reclaim_producer_cleanup(namespace, context=CONTEXT, limit=32)
        assert replay.removed == 0 and not replay.remaining
        assert len(provider.requests) == 1
    finally:
        policy.denied.clear()
        policy.allowed = None
        await app.drain_collaboration_requests()
