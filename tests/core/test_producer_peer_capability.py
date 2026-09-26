"""A numerically equal value is not a supported receiving protocol version."""

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.providers.base import ModelStreamEvent


@pytest.mark.anyio
async def test_public_producer_refuses_malformed_receiving_version_before_dispatch(
    native_stores, monkeypatch
):
    app, resolver, _, provider, session, _, command, execution = await output_scenario(
        native_stores, with_exports=True
    )
    app._request_coordinator._owners.observation_timeout = 60
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
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

    async def execute():
        return [
            event
            async for event in app.execute_producer_output(
                command, execution, context=CONTEXT, producer_context=resolver.recipient.context
            )
        ]

    proposal = ProducerOutputProposal(
        operation=command.operation,
        admission=command.admission,
        binding_incarnation=command.binding_incarnation,
        limits=command.limits,
        destinations=command.destinations,
    )
    try:
        for capability in ("peer_content_version", "session_steering_version"):
            for version in (True, 1.0, "1", None, 2):
                with monkeypatch.context() as fault:
                    fault.setattr(native_stores[1], capability, version)
                    if capability == "session_steering_version":
                        with pytest.raises(CollaborationUnavailable):
                            await app.prepare_producer_output(
                                proposal, execution, context=resolver.recipient.context
                            )
                        with pytest.raises(CollaborationUnavailable):
                            await app.register_producer_output(
                                command, execution, context=resolver.recipient.context
                            )
                    with pytest.raises(CollaborationUnavailable):
                        await execute()
                assert provider.requests == []
                assert (await native_stores[1].load(session.id)).run_epoch == 0
        provider._batches = (
            (ModelStreamEvent.text_delta("qualified"), ModelStreamEvent.completed()),
        )
        await execute()
        assert len(provider.requests) == 1
        completion = await app.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == "answer"
    finally:
        await app.drain_collaboration_requests()
