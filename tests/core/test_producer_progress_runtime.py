"""Producer milestones reflect actual native invocation admission and publication."""

import asyncio
from contextlib import aclosing

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerProgressOccurrence
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.providers.base import ModelStreamEvent


@pytest.mark.anyio
async def test_running_producer_progress_observes_actual_invocation(native_stores, monkeypatch):
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
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    original = resolver.recipient.resolution
    actions = tuple(dict.fromkeys((*original.principal.actions, "execute", "publish")))
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
    context = resolver.recipient.context

    async def milestone(sequence, kind):
        current = await app.inspect_collaboration_request(admission.expected, context=context)
        occurrence = ProducerProgressOccurrence(
            operation=initialized.operation("native-milestone-" + str(sequence)),
            expected_revision=current.revision,
            sequence=sequence,
            kind=kind,
        )
        receipt = await app.record_producer_progress(command, occurrence, context=context)
        assert await app.record_producer_progress(command, occurrence, context=context) == receipt
        return receipt

    prepared = await milestone(1, "prepared")
    assert prepared.command.evidence.run_epoch is None
    entered, release = asyncio.Event(), asyncio.Event()
    stream = provider.stream
    provider._batches = ((ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()),)

    async def blocked(request):
        entered.set()
        await release.wait()
        async with aclosing(stream(request)) as events:
            async for event in events:
                yield event

    monkeypatch.setattr(provider, "stream", blocked)

    async def consume():
        async with aclosing(
            app.execute_producer_output(
                command, execution, context=CONTEXT, producer_context=context
            )
        ) as events:
            return [event async for event in events]

    observer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 60)
        started = await milestone(2, "started")
        producing = await milestone(3, "producing")
        assert started.command.evidence.run_epoch == producing.command.evidence.run_epoch == 1
        assert started.command.evidence.interaction_id == producing.command.evidence.interaction_id
        assert (await app.session_store.load(session.id)).status == "running"
        release.set()
        events = await asyncio.wait_for(observer, 90)
        assert any(event.type == "session.completed" for event in events)
        published = await milestone(4, "published")
        assert (
            published.command.evidence.native_commitment
            != producing.command.evidence.native_commitment
        )
        with pytest.raises(CollaborationUnavailable):
            await milestone(5, "producing")
        current = await app.inspect_collaboration_request(admission.expected, context=context)
        assert tuple(item.operation for item in current.progress) == tuple(
            item.command.operation for item in (prepared, started, producing, published)
        )
        assert len(provider.requests) == 1
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await app.drain_collaboration_requests()
