"""Caller cancellation leaves native progress publication owned and exactly replayable."""

import asyncio

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ProducerProgressOccurrence


@pytest.mark.anyio
async def test_cancelled_progress_observer_preserves_late_commit_and_exact_replay(
    native_stores, monkeypatch
):
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, _, initialized, registration, _ = values
    context = resolver.recipient.context
    prior = await app.inspect_collaboration_request(admission.expected, context=context)
    occurrence = ProducerProgressOccurrence(
        operation=initialized.operation("cancelled-progress-observer"),
        expected_revision=prior.revision,
        sequence=1,
        kind="published",
    )
    receiver = app._request_coordinator._registration.receiving_owner
    read = receiver._read_producer_progress
    entered, release = asyncio.Event(), asyncio.Event()
    reads = []

    async def blocked(*args, **kwargs):
        result = await read(*args, **kwargs)
        reads.append(result)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(receiver, "_read_producer_progress", blocked)
    observer = asyncio.create_task(
        app.record_producer_progress(registration, occurrence, context=context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 60)
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(observer, 10)
        assert observer.cancelled() and observer.cancelling() == 2
        pending = tuple(app._request_coordinator._owners.pending)
        assert pending
        before = await app.inspect_collaboration_request(admission.expected, context=context)
        assert before.progress == ()
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending), 60)
        assert not app._request_coordinator._owners.pending
        # Historical acknowledgement recovery needs readback, not a new
        # publication grant or another native observation.
        resolution = resolver.recipient.resolution
        resolver.recipient.resolution = resolution.model_copy(
            update={
                "principal": resolution.principal.model_copy(
                    update={
                        "actions": tuple(
                            action for action in resolution.principal.actions if action != "publish"
                        ),
                    }
                ),
            }
        )

        async def forbidden(*args, **kwargs):
            raise AssertionError("Exact progress replay must not reread native production")

        monkeypatch.setattr(receiver, "_read_producer_progress", forbidden)
        first = await app.record_producer_progress(registration, occurrence, context=context)
        assert (
            await app.record_producer_progress(registration, occurrence, context=context) == first
        )
        after = await app.inspect_collaboration_request(admission.expected, context=context)
        assert len(after.progress) == 1 and after.progress[0].operation == occurrence.operation
        assert len(reads) == len(provider.requests) == 1
    finally:
        release.set()
        if not observer.done():
            observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        await app.drain_collaboration_requests()
