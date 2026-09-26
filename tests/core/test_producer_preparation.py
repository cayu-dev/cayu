"""Public producer preparation derives native identity without acquiring authority."""

import warnings

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal
from cayu.collaboration._contracts import CollaborationConflict, ExactNotFound
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.messages import Message
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


@pytest.mark.anyio
async def test_public_preparation_derives_exact_native_identity_without_registration(
    native_stores, monkeypatch, caplog, capsys
):
    values = await output_scenario(native_stores)
    app, resolver, _, provider, session, _, expected, execution = values
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    proposal = ProducerOutputProposal(
        operation=expected.operation,
        admission=expected.admission,
        binding_incarnation=expected.binding_incarnation,
        limits=expected.limits,
        destinations=expected.destinations,
    )
    context = resolver.recipient.context
    prepared = await app.prepare_producer_output(proposal, execution, context=context)
    assert prepared == expected
    assert await app.prepare_producer_output(proposal, execution, context=context) == prepared
    assert isinstance(
        await app.lookup_producer_registration(prepared, context=CONTEXT), ExactNotFound
    )
    assert not provider.requests
    assert await native_stores[1].load(session.id) == session

    changed = ParticipantSessionExecutionRequest(
        request=execution.request.model_copy(update={"messages": [Message.text("user", "other")]}),
        session_instance_id=execution.session_instance_id,
        execution_key=execution.execution_key,
    )
    with pytest.raises(CollaborationUnavailable):
        await app.prepare_producer_output(proposal, changed, context=context)
    with pytest.raises(CollaborationConflict):
        await app.prepare_producer_output(
            proposal,
            execution,
            context=context.model_copy(update={"participant": proposal.destinations[0].recipient}),
        )
    assert isinstance(
        await app.lookup_producer_registration(prepared, context=CONTEXT), ExactNotFound
    )

    canary = "rejected-producer-proposal-secret-canary"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    malformed = proposal.model_copy(update={"binding_incarnation": Hostile()})
    caplog.clear()
    capsys.readouterr()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as error:
            await app.prepare_producer_output(malformed, execution, context=context)
    output = capsys.readouterr()
    assert not captured
    assert canary not in output.out + output.err + caplog.text + str(error.value) + repr(
        error.value
    )

    registered = await app.register_producer_output(prepared, execution, context=context)
    assert registered.command == prepared
    assert not provider.requests
