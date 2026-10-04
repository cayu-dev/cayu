"""Independent publication owners reconcile a lost commit acknowledgement exactly."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import app as make_app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ProducerProgressOccurrence
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration


@pytest.mark.anyio
async def test_progress_independent_workers_reconcile_commit_ack_loss(native_stores, monkeypatch):
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, session, initialized, registration, _ = values
    context = resolver.recipient.context

    def replacement(*, with_native):
        return make_app(
            native_stores[2](),
            app._participant_coordinator._registration,
            **({"session_store": app.session_store} if with_native else {}),
            collaboration_requests=RequestRegistration(
                mandates=resolver,
                prepared_admission=PreparedAdmissionRegistration(receiver=registration.receiver),
                max_ttl_ms=300_000,
            ),
        )

    other = replacement(with_native=True)
    await other.initialize_collaboration()
    monkeypatch.setattr(other._request_coordinator._owners, "observation_timeout", 60)
    prior = await app.inspect_collaboration_request(admission.expected, context=context)
    occurrence = ProducerProgressOccurrence(
        operation=initialized.operation("concurrent-native-progress"),
        expected_revision=prior.revision,
        sequence=1,
        kind="published",
    )
    original_transaction = native_stores[0]._transaction
    lost = []

    @asynccontextmanager
    async def lose_ack(scope, *, write):
        published = False
        async with original_transaction(scope, write=write) as tx:
            put = tx.put

            async def track(family, key, value, *, insert):
                nonlocal published
                await put(family, key, value, insert=insert)
                published |= family == "operations" and key == operation_key(occurrence.operation)

            tx.put = track
            yield tx
        if published and not lost:
            lost.append(True)
            raise ConnectionError("Progress commit acknowledgement lost")

    # Only the first worker loses its commit ACK. Ensure it commits first while
    # the independent second worker has already observed the same absent key.
    entered, release = asyncio.Event(), asyncio.Event()
    receiver = other._request_coordinator._registration.receiving_owner
    read = receiver._read_producer_progress

    async def blocked(*args, **kwargs):
        result = await read(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(receiver, "_read_producer_progress", blocked)
    competing = asyncio.create_task(
        other.record_producer_progress(registration, occurrence, context=context)
    )
    recovered = None
    try:
        await asyncio.wait_for(entered.wait(), 60)
        with monkeypatch.context() as patch:
            patch.setattr(native_stores[0], "_transaction", lose_ack)
            with pytest.raises(CollaborationUnavailable):
                await app.record_producer_progress(registration, occurrence, context=context)
        assert lost == [True]
        release.set()
        first = await asyncio.wait_for(competing, 60)
        assert (
            await app.record_producer_progress(registration, occurrence, context=context) == first
        )
        after = await app.inspect_collaboration_request(admission.expected, context=context)
        assert after.revision == prior.revision + 1 and len(after.progress) == 1
        assert after.progress[0].operation == occurrence.operation

        recovered = replacement(with_native=False)
        await recovered.initialize_collaboration()
        monkeypatch.setattr(recovered._request_coordinator._owners, "observation_timeout", 60)
        assert not recovered._provider_registry.registrations
        assert await recovered.session_store.load(session.id) is None
        assert (
            await recovered.record_producer_progress(registration, occurrence, context=context)
            == first
        )
        with pytest.raises(CollaborationConflict):
            await recovered.record_producer_progress(
                registration, occurrence.model_copy(update={"kind": "started"}), context=context
            )
        assert len(provider.requests) == 1
    finally:
        release.set()
        if not competing.done():
            competing.cancel()
        await asyncio.gather(competing, return_exceptions=True)
        await other.drain_collaboration_requests()
        if recovered is not None:
            await recovered.drain_collaboration_requests()
