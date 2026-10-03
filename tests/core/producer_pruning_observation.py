"""Observe retained maintenance work, then require exact public receipt replay."""

import asyncio

from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration.participants import CollaborationUnavailable


async def prune_to_receipt(app, batch, *, pending_observed=None):
    return await _maintenance_to_receipt(
        app, app.prune_collaboration_namespace, batch, pending_observed=pending_observed
    )


async def retire_to_receipt(app, request, *, pending_observed=None):
    return await _maintenance_to_receipt(
        app, app.retire_collaboration_namespace, request, pending_observed=pending_observed
    )


async def _maintenance_to_receipt(app, operation, request, *, pending_observed):
    try:
        return await operation(request, context=CONTEXT)
    except CollaborationUnavailable:
        store, _ = app._participant_coordinator._ready()
        pending = tuple(store._owners.pending)
        if not pending:
            raise
        # This fixture has one maintenance operation. An acknowledgement timeout
        # is not mutation failure, but neither pending state nor a retry alone
        # establishes success. Require the original owner to return its receipt.
        assert len(pending) == 1
        if pending_observed is not None:
            pending_observed.set()
        original = await asyncio.wait_for(asyncio.shield(pending[0]), 60)
        receipt = await operation(request, context=CONTEXT)
        assert receipt == original
        return receipt
