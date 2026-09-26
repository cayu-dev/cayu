"""Request-only production cannot infer an independent obligation from detach."""

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal


@pytest.mark.anyio
@pytest.mark.parametrize("planned", [False, True])
async def test_public_request_only_producer_refuses_detach(native_stores, monkeypatch, planned):
    app, resolver, admission, provider, session, _, command, execution = await output_scenario(
        native_stores, planned=planned, cancellation="detach", unchecked_registration=True
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    proposal = ProducerOutputProposal(
        operation=command.operation,
        admission=admission,
        binding_incarnation=command.binding_incarnation,
        limits=command.limits,
        destinations=command.destinations,
    )
    participant = await app.inspect_participant(admission.prepared.recipient, context=CONTEXT)
    with pytest.raises(ValueError):
        await app.prepare_producer_output(proposal, execution, context=resolver.recipient.context)
    with pytest.raises(ValueError):
        await app.register_producer_output(command, execution, context=resolver.recipient.context)
    with pytest.raises(ValueError):
        async for _ in app.execute_producer_output(
            command, execution, context=CONTEXT, producer_context=resolver.recipient.context
        ):
            pytest.fail("Unqualified detached producer reached execution")
    assert not provider.requests
    assert await native_stores[1].load(session.id) == session
    assert (
        await app.inspect_participant(admission.prepared.recipient, context=CONTEXT)
    ) == participant
    assert not (
        await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    ).items
