"""Assertions for the real finite-broadcast cleanup handoff."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration.exports import SessionExportConflict, SessionExportSettlementRequest
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.sessions import RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest


async def settle_broadcast(app, proposal, resolver, context, completion, provider, *, deliveries):
    """No partial receiving settlement authorizes releasing native retention."""
    resolution = resolver.recipient.resolution
    actions = tuple(dict.fromkeys((*resolution.principal.actions, "release", "retire")))
    resolver.recipient.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(update={"actions": actions}),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in resolution.chain.entries
                    )
                }
            ),
        }
    )
    store, initialized = app._participant_coordinator._ready()
    target = proposal.admission.prepared.target
    for destination, delivery in zip(proposal.destinations, deliveries, strict=True):
        if delivery is None:
            retired = await app.retire_producer_export(
                proposal, destination.operation, context=CONTEXT
            )
            assert retired.state == "retired"
            continue
        assert delivery.destination == destination.operation
        with pytest.raises(CollaborationUnavailable):
            await app.settle_producer_output(proposal, context=CONTEXT)
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            pending = await read_output_registration(tx, proposal, redactor=app._secret_redactor)
        assert pending.cleanup is None and pending.cleanup_ack is None
        assert pending.reserved_operations > 0 and pending.reserved_bytes > 0
        with pytest.raises((SessionExportConflict, ValueError)) as blocked:
            await app.session_store.delete_session(target.session_id)
        if not isinstance(blocked.value, SessionExportConflict):
            assert type(blocked.value) is ValueError
            assert (
                str(blocked.value)
                == "Session closure requires settled producer output responsibility."
            )
        exported = await app.export_producer_output(
            proposal, destination.operation, context=context
        )
        request = SessionExportSettlementRequest(
            request=exported.request,
            mode="release" if delivery.receipt.status == "appended" else "retire",
            operation=exported.request.ref.operation.model_copy(
                update={"caller_key": "release:" + destination.operation.caller_key}
            ),
        )
        receipt = await app.settle_session_export(request, context=context)
        assert (receipt.acceptance is not None) == (delivery.receipt.status == "appended")
        assert await app.settle_session_export(request, context=context) == receipt
    try:
        final = await app.settle_producer_output(proposal, context=CONTEXT)
    except CollaborationUnavailable:
        owners = tuple(app._request_coordinator._owners.pending)
        if not owners:
            raise
        # The public foreground deadline is an observation bound, not a claim
        # that a multi-destination cleanup transaction stopped. Finish observing
        # its retained owner, then reconcile the SAME operation and original key.
        outcomes = await asyncio.wait_for(asyncio.gather(*owners, return_exceptions=True), 180)
        assert not any(isinstance(outcome, BaseException) for outcome in outcomes), outcomes
        final = await app.settle_producer_output(proposal, context=CONTEXT)
    assert final.delivery == "published"
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        settled = await read_output_registration(tx, proposal, redactor=app._secret_redactor)
    assert tuple(item.destination for item in settled.cleanup.evidence.destinations) == tuple(
        destination.operation for destination in proposal.destinations
    )
    assert settled.cleanup_ack == final.operation
    assert (settled.reserved_operations, settled.reserved_events, settled.reserved_bytes) == (
        0,
        0,
        0,
    )
    await app.session_store.delete_session(target.session_id)
    # Public-ID reuse cannot redirect an old producer cleanup to new work.
    replacement, _ = await app.create_recipient_session(
        RecipientSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", session_id=target.session_id, messages=[]),
            creation_key="replacement-after-producer-cleanup:"
            + proposal.operation.application_scope,
            recipient=proposal.admission.prepared.recipient,
        ),
        context=CONTEXT,
    )
    assert replacement.id == target.session_id
    assert replacement.instance_id != target.session_instance_id
    assert await app.settle_producer_output(proposal, context=CONTEXT) == final
    assert await app.retain_producer_completion(proposal, context=CONTEXT) == completion
    assert await app.session_store.load(replacement.id) == replacement
    assert len(provider.requests) == 1
