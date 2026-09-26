"""Closure after native completion uses release evidence, not a fictitious stop."""

import pytest
from tests.core import test_prepared_admission_public as preparations
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import ExactMatch, ExactNotFound
from cayu.collaboration._producer_completion import retain_producer_completion
from cayu.collaboration._request_store import retained_request
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest


@pytest.mark.anyio
async def test_closure_after_native_completion_reconciles_exact_release(native_stores, monkeypatch):
    setup = preparations.setup

    async def stop_on_closure(store):
        values = list(await setup(store))
        values[4] = values[4].model_copy(update={"cancellation": "stop"})
        return tuple(values)

    monkeypatch.setattr(preparations, "setup", stop_on_closure)
    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(native_stores, with_exports=True, planned=True)
    assert proposal.disposition == "stop"
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(proposal, execution, context=resolver.recipient.context)
    original = resolver.recipient.resolution
    actions = (*original.principal.actions, "execute")
    resolver.recipient.resolution = original.model_copy(
        update={
            "principal": original.principal.model_copy(update={"actions": actions}),
            "chain": original.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in original.chain.entries
                    )
                }
            ),
        }
    )
    provider._batches = (
        (ModelStreamEvent.text_delta("completed answer"), ModelStreamEvent.completed()),
    )
    async for _ in app.execute_producer_output(
        proposal,
        execution,
        context=CONTEXT,
        producer_context=resolver.recipient.context,
    ):
        pass
    assert len(provider.requests) == 1
    assert (await native_stores[1].load(session.id)).status == "completed"
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        snapshot = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            app._secret_redactor,
        )
    assert snapshot is not None
    request = RequestControl(
        operation=initialized.operation("close-completed-producer"),
        expected=admission.expected,
        expected_revision=snapshot.revision,
        kind="cancel",
    )
    receiver = app._request_coordinator._registration.receiving_owner

    async def missing_release(*args):
        raise ValueError("Exact native release evidence is unavailable.")

    with monkeypatch.context() as patch:
        patch.setattr(receiver, "_read_producer_release", missing_release)
        with pytest.raises(CollaborationUnavailable):
            await app.control_collaboration_request(request, context=resolver.sender.context)
    # The request may close, but status alone must not discharge responsibility.
    pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    assert any(item.recovery.registration == proposal.operation for item in pending.items)
    control = await app.control_collaboration_request(request, context=resolver.sender.context)
    status = await app.service_producer_disposition(proposal, context=CONTEXT)
    assert status.disposition == "invocation_released"
    assert status.native_stop is None and status.native_release_commitment is not None
    assert await app.service_producer_disposition(proposal, context=CONTEXT) == status
    assert (
        await app.control_collaboration_request(request, context=resolver.sender.context) == control
    )
    pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    assert any(item.recovery.registration == proposal.operation for item in pending.items)
    assert isinstance(
        await app.lookup_producer_completion(proposal, context=CONTEXT), ExactNotFound
    )
    policy = app._participant_coordinator._registration.access_policy
    policy.denied.add("request_control")
    with pytest.raises(CollaborationAccessDenied):
        await app.retain_producer_completion(proposal, context=CONTEXT)
    with pytest.raises(CollaborationAccessDenied):
        await app.settle_producer_output(proposal, context=CONTEXT)
    assert isinstance(
        await app.lookup_producer_completion(proposal, context=CONTEXT), ExactNotFound
    )
    # Mandatory owner-internal retention is not disabled by revoking public
    # control. The identical caller-visible command alone grants no such access.
    completion = await retain_producer_completion(app, proposal)
    policy.denied.clear()
    assert await app.retain_producer_completion(proposal, context=CONTEXT) == completion
    found = await app.lookup_producer_completion(proposal, context=CONTEXT)
    assert isinstance(found, ExactMatch) and found.receipt == completion
    assert completion.output.disposition == "answer"
    final = await app.settle_producer_output(proposal, context=CONTEXT)
    assert final.delivery == "excluded"
    assert len(provider.requests) == 1
    await native_stores[1].delete_session(session.id)
    assert await app.settle_producer_output(proposal, context=CONTEXT) == final
    found = await app.lookup_producer_completion(proposal, context=CONTEXT)
    assert isinstance(found, ExactMatch) and found.receipt == completion
    assert (
        await app.control_collaboration_request(request, context=resolver.sender.context) == control
    )
    replacement, _ = await app.create_recipient_session(
        RecipientSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", session_id=session.id, messages=[]),
            creation_key="replacement-after-closed-producer",
            recipient=admission.prepared.recipient,
        ),
        context=CONTEXT,
    )
    assert replacement.id == session.id and replacement.instance_id != session.instance_id
    assert (
        await app.control_collaboration_request(request, context=resolver.sender.context) == control
    )
    assert await native_stores[1].load(replacement.id) == replacement
    assert len(provider.requests) == 1
