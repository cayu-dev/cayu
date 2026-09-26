"""Real observer loss after native safe-stop acceptance, before source return."""

import asyncio

import pytest

from cayu.collaboration._request_store import retained_request
from cayu.collaboration.participants import CollaborationUnavailable


async def close_with_native_observation_failure(app, request, context, failure, monkeypatch):
    if failure == "ordinary":
        return await app.control_collaboration_request(request, context=context)
    receiver = app._request_coordinator._registration.receiving_owner
    stop = receiver._request_producer_stop
    entered, release = asyncio.Event(), asyncio.Event()
    accepted = []

    async def lose_ack(*args, **kwargs):
        receipt = await stop(*args, **kwargs)
        accepted.append(receipt)
        if len(accepted) == 1:
            if failure == "ack":
                raise RuntimeError("native stop acknowledgement lost after commit")
            entered.set()
            await release.wait()
        return receipt

    foreground = None
    with monkeypatch.context() as patch:
        patch.setattr(receiver, "_request_producer_stop", lose_ack)
        try:
            if failure == "ack":
                with pytest.raises(CollaborationUnavailable):
                    await app.control_collaboration_request(request, context=context)
            else:
                assert failure == "cancel"
                foreground = asyncio.create_task(
                    app.control_collaboration_request(request, context=context)
                )
                await asyncio.wait_for(entered.wait(), 60)
                foreground.cancel()
                foreground.cancel()
                assert foreground.cancelling() == 2
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(foreground, 5)
                assert foreground.cancelled() and foreground.cancelling() == 2
                assert app._request_coordinator._owners.pending
            assert len(accepted) == 1
            store, initialized = app._participant_coordinator._ready()
            async with store._transaction(initialized.owner.application_scope, write=False) as tx:
                retained = await retained_request(
                    store,
                    tx,
                    initialized,
                    request.expected.intent.request,
                    request.expected.initiator,
                    app._secret_redactor,
                )
            assert retained is not None and retained.state == "cancelled"
            assert retained.producer_settlement is None
            assert retained.terminal is not None
            assert retained.terminal.expected.operation == request.operation
            owners = tuple(app._request_coordinator._owners.pending)
            release.set()
            if owners:
                await asyncio.wait_for(asyncio.gather(*owners), 60)
            replay = await app.control_collaboration_request(request, context=context)
            assert replay == retained.terminal
            assert len(accepted) == 2 and accepted[0] == accepted[1]
            return replay
        finally:
            release.set()
            if foreground is not None:
                if not foreground.done():
                    foreground.cancel()
                await asyncio.gather(foreground, return_exceptions=True)
