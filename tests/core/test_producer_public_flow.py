"""Public registration/execution/export/election/peer delivery reaches a real model request."""

from contextlib import aclosing, asynccontextmanager

import pytest
from tests.core import test_peer_content as peer_fixtures
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration import ProducerDeliveryRecovery
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._producer_outcome_store import outcome_operation
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentExposureRequest
from cayu.providers.openai import OpenAIProvider
from cayu.sessions import RunRequest
from cayu.sessions.context_views import ParticipantSessionExecutionRequest
from cayu.sessions.invocation import InvocationOriginClaim


@pytest.mark.anyio
@pytest.mark.parametrize("planned", [False, True])
async def test_public_producer_output_reaches_recipient_provider_without_private_parts(
    native_stores, monkeypatch, planned
):
    private = "private-producer-thinking-not-an-answer"
    payloads = []

    class Transport:
        async def create_response(self, **kwargs):
            payloads.append(kwargs["payload"])
            return {
                "id": "qualified-receiving-response",
                "status": "completed",
                "model": "model",
                "output": [
                    {
                        "id": "qualified-receiving-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "received review", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }

    class Provider(peer_fixtures.QualifiedPeerProvider):
        async def stream(self, request):
            if self.requests:
                self.requests.append(request)
                adapter = OpenAIProvider(api_key="test-key", streaming=False, transport=Transport())
                async with aclosing(adapter.stream(request)) as stream:
                    async for event in stream:
                        yield event
            else:
                async with aclosing(super().stream(request)) as stream:
                    async for event in stream:
                        yield event

    monkeypatch.setattr(peer_fixtures, "QualifiedPeerProvider", Provider)
    origin = InvocationOriginClaim(subject=CONTEXT.principal)
    values, context = await completed_export_scenario(
        native_stores, monkeypatch, private_text=private, consumer_origin=origin, planned=planned
    )
    app, _resolver, _, provider, _session, _, proposal, _ = values
    destination = proposal.destinations[0]
    exported = await app.export_producer_output(proposal, destination.operation, context=context)
    transaction = native_stores[0]._transaction
    outcome_key = operation_key(outcome_operation(proposal, app._secret_redactor))
    lost = []

    @asynccontextmanager
    async def lose_election_ack(scope, *, write):
        committed = False
        async with transaction(scope, write=write) as tx:
            yield tx
            if write and not lost:
                committed = await tx.get("operations", outcome_key) is not None
        if committed:
            lost.append(True)
            raise ConnectionError("Outcome election committed; acknowledgement lost")

    with monkeypatch.context() as failure:
        failure.setattr(native_stores[0], "_transaction", lose_election_ack)
        with pytest.raises(CollaborationUnavailable):
            await app.publish_producer_outcome(
                proposal, destination=destination.operation, context=context
            )
    assert lost == [True] and len(provider.requests) == 1
    elected = await app.publish_producer_outcome(
        proposal, destination=destination.operation, context=context
    )
    assert (
        await app.publish_producer_outcome(
            proposal, destination=destination.operation, context=context
        )
        == elected
    )
    assert elected.command.outcome == "answered"
    source = await app.lookup_session_export(exported.request, context=context)
    assert isinstance(source, ExactMatch)
    policy = app._session_export_coordinator.registration.policy
    policy.register_export(
        source.receipt,
        payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
        consumer_id=destination.recipient.participant_id,
    )
    policy.allowed_receipts.add(proposal.operation.caller_key)
    policy.denied.add("append")
    with pytest.raises((CollaborationUnavailable, CollaborationAccessDenied)):
        await app.deliver_producer_output(proposal, destination.operation, context=context)
    assert len(provider.requests) == 1
    policy.denied.clear()
    delivered = await app.deliver_producer_output(proposal, destination.operation, context=context)
    assert delivered.receipt is not None and delivered.receipt.status == "appended"
    assert delivered.receipt.occurrence.payload.text == "retained answer"
    assert private not in delivered.model_dump_json()
    assert (
        await app.deliver_producer_output(proposal, destination.operation, context=context)
        == delivered
    )
    pending = await app.pending_producer_outputs(
        proposal.admission.prepared.recipient, context=CONTEXT
    )
    token = next(
        item.recovery for item in pending.items if item.recovery.registration == proposal.operation
    )
    recovery = ProducerDeliveryRecovery(
        registration=token.registration,
        registration_commitment=token.registration_commitment,
        destination=destination.operation,
    )
    recovered = await app.reconcile_producer_delivery(recovery, context=CONTEXT)
    assert recovered.state == "appended"
    # A caller cannot change an already committed append into an exclusion.
    assert (
        await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True) == recovered
    )
    target = destination.attempt.append_key
    execution = ParticipantSessionExecutionRequest(
        request=RunRequest(
            agent_name="reviewer",
            session_id=target.target_session_id,
            messages=[],
            invocation_origin=origin,
        ),
        session_instance_id=target.target_session_instance_id,
        execution_key="consume-retained-producer-answer",
    )
    events = []
    async for event in app.execute_participant_session(
        execution, participant=destination.recipient, context=CONTEXT
    ):
        events.append(event)
    if len(provider.requests) != 2:
        import json

        pytest.fail(
            "Recipient did not reach the provider: "
            + json.dumps([(event.type, event.payload) for event in events[-4:]])
        )
    assert any(event.type == "session.completed" for event in events)
    assert not any(event.type == "session.failed" for event in events)
    observed = provider.requests[-1].model_dump_json()
    assert "retained answer" in observed
    assert private not in observed
    assert len(payloads) == 1 and "retained answer" in repr(payloads[0])
    assert private not in repr(payloads[0])
    assert app._session_export_coordinator.projectors[destination.projector].calls == 1
    assert len(policy.calls) == 1
    call = policy.calls[0]
    exposure = PeerContentExposureRequest.for_model_attempt(
        append_key=delivered.append.append_key,
        append_operation_key=delivered.append.operation_key,
        model_attempt_id=call["model_attempt_id"],
        provider_name=call["provider_name"],
        capability_version=call["capability_version"],
    )
    retained_exposure = await app.session_store.read_peer_content_exposure(
        exposure.append_key, exposure.exposure_id
    )
    assert retained_exposure is not None and retained_exposure.outcome == "exposed"
    # Consumer execution is distinct from the single completed producer invocation.
    assert (
        await app.publish_producer_outcome(
            proposal, destination=destination.operation, context=context
        )
        == elected
    )
    assert len(provider.requests) == 2
