"""Closing the public producer stream retains exact post-dispatch responsibility."""

import asyncio
from contextlib import aclosing

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.storage import _producer_observation


@pytest.mark.anyio
@pytest.mark.parametrize("termination", ["close", "cancel"])
async def test_public_producer_stream_close_after_dispatch_keeps_cleanup_owned(
    native_stores, monkeypatch, termination
):
    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        command,
        execution,
    ) = await output_scenario(native_stores, with_exports=True)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    resolution = resolver.recipient.resolution
    actions = (*resolution.principal.actions, "execute")
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
    provider._batches = (
        (ModelStreamEvent.text_delta("partial answer"), ModelStreamEvent.completed()),
    )
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_stream = provider.stream

    async def blocked_provider(request):
        try:
            async with aclosing(original_stream(request)) as events:
                async for event in events:
                    yield event
                    if event.type == "text_delta":
                        entered.set()
                        await release.wait()
        finally:
            stopped.set()

    if termination == "cancel":
        monkeypatch.setattr(provider, "stream", blocked_provider)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    stream = app.execute_producer_output(
        command, execution, context=CONTEXT, producer_context=resolver.recipient.context
    )
    task = None
    cancellations = []

    async def consume():
        try:
            async with aclosing(stream) as events:
                async for _ in events:
                    pass
        except asyncio.CancelledError:
            cancellations.append(True)
            raise

    try:
        if termination == "cancel":
            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), 90)
            task.cancel()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 90)
            assert task.cancelled() and task.cancelling() == 2 and cancellations == [True]
            assert stopped.is_set()
        else:
            async with asyncio.timeout(90):
                async for event in stream:
                    if event.type is EventType.MODEL_TEXT_DELTA:
                        break
                else:
                    pytest.fail("The producer never reached an actual provider text event")
        assert len(provider.requests) == 1
        await asyncio.wait_for(stream.aclose(), 90)
        pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        assert any(item.recovery.registration == command.operation for item in pending.items)
        snapshot = await app.inspect_collaboration_request(
            admission.expected, context=resolver.sender.context
        )
        assert snapshot.outcome is None and snapshot.producer_settlement is None
        native_release = await native_stores[1]._read_native_producer_release(command)
        before = await app.lookup_producer_completion(command, context=CONTEXT)

        observations = []

        def unavailable_release(*args):
            observations.append(True)
            raise ValueError("Native release receipt is unavailable")

        def foreign_release(*args):
            observations.append(True)
            return native_release.model_copy(update={"interaction_id": "another-invocation"})

        for replacement in (unavailable_release, foreign_release):
            with monkeypatch.context() as patch:
                patch.setattr(_producer_observation, "_project", replacement)
                with pytest.raises(CollaborationUnavailable):
                    await app.retain_producer_completion(command, context=CONTEXT)
            assert await app.lookup_producer_completion(command, context=CONTEXT) == before
        assert len(observations) == 2
        await app.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("close-abandoned-producer"),
                expected=admission.expected,
                expected_revision=snapshot.revision,
                kind="cancel",
            ),
            context=resolver.sender.context,
        )
        completion = await app.retain_producer_completion(command, context=CONTEXT)
        assert completion.output.disposition == "stopped"
        settled = await app.settle_producer_output(command, context=CONTEXT)
        assert settled.delivery == "excluded"
        assert await app.settle_producer_output(command, context=CONTEXT) == settled
        assert not (
            await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        ).items
        await native_stores[1].delete_session(session.id)
        assert await app.retain_producer_completion(command, context=CONTEXT) == completion
        assert len(provider.requests) == 1
    finally:
        release.set()
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await stream.aclose()
        await app.drain_collaboration_requests()
