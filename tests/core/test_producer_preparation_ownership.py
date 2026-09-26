"""Preparation snapshots mutable input and retains opaque observation ownership."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerOutputProposal
from cayu.collaboration import _producer_preparation as preparation
from cayu.collaboration._contracts import ExactNotFound
from cayu.messages import Message
from cayu.sessions.context_views import ParticipantSessionExecutionRequest


@pytest.mark.anyio
@pytest.mark.parametrize("observation", ["return", "cancel", "deadline"])
async def test_preparation_snapshot_and_cancelled_observer_remain_non_dispatching(
    native_stores, monkeypatch, observation
):
    app, resolver, _, provider, session, _, expected, execution = await output_scenario(
        native_stores, planned=True
    )
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    proposal = ProducerOutputProposal(
        operation=expected.operation,
        admission=expected.admission,
        binding_incarnation=expected.binding_incarnation,
        limits=expected.limits,
        destinations=expected.destinations,
    )
    original = ParticipantSessionExecutionRequest(
        request=execution.request,
        session_instance_id=execution.session_instance_id,
        execution_key=execution.execution_key,
    )
    native = preparation._native_execution_commitment
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()
        return await native(*args)

    monkeypatch.setattr(preparation, "_native_execution_commitment", blocked)
    observer = asyncio.create_task(
        app.prepare_producer_output(proposal, execution, context=resolver.recipient.context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 60)
        execution.request.messages.append(Message.text("user", "late caller mutation"))
        assert original.request.messages != execution.request.messages
        if observation == "cancel":
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(observer, 10)
            assert observer.cancelled() and observer.cancelling() == 2
            assert app._request_coordinator._owners.pending
        elif observation == "deadline":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(observer, 0.01)
            assert observer.cancelled() and observer.cancelling() == 1
            assert app._request_coordinator._owners.pending
        release.set()
        if observation == "return":
            assert await asyncio.wait_for(observer, 60) == expected
        # Public drain is terminal for this owner. Observe already-retained
        # work here, then exercise exact retry before terminal shutdown below.
        await asyncio.wait_for(asyncio.gather(*tuple(app._request_coordinator._owners.pending)), 60)
        assert not app._request_coordinator._owners.pending
        assert isinstance(
            await app.lookup_producer_registration(expected, context=CONTEXT), ExactNotFound
        )
        assert not provider.requests
        assert await native_stores[1].load(session.id) == session
        assert (
            await app.prepare_producer_output(
                proposal, original, context=resolver.recipient.context
            )
            == expected
        )
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await app.drain_collaboration_requests()
