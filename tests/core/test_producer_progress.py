"""Native FRESH progress uses common request arbitration without caller authority."""

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import ProducerProgressOccurrence
from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration._request_arbitration import progress_in_transaction
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
async def test_native_progress_replays_and_fixes_terminal_frontier(native_stores, monkeypatch):
    values, disclosure = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, _, initialized, registration, _ = values
    context = resolver.recipient.context
    prior = await app.inspect_collaboration_request(admission.expected, context=context)
    occurrence = ProducerProgressOccurrence(
        operation=initialized.operation("native-progress-one"),
        expected_revision=prior.revision,
        sequence=1,
        kind="published",
    )
    first = await app.record_producer_progress(registration, occurrence, context=context)
    assert first.command.evidence.registration == registration.operation
    assert first.command.evidence.run_epoch == 1
    assert first.command.source_receipt is None
    assert await app.record_producer_progress(registration, occurrence, context=context) == first
    retained = await app.inspect_collaboration_request(admission.expected, context=context)
    key = operation_key(admission.expected.operation)
    corruptions = (
        retained.model_copy(update={"progress": ()}),
        retained.model_copy(
            update={
                "progress": (
                    retained.progress[0].model_copy(
                        update={
                            "receipt_commitment": "sha256:" + "0" * 64,
                        }
                    ),
                )
            }
        ),
    )
    for corrupted in corruptions:
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=True
        ) as tx:
            await tx.put("requests", key, corrupted, insert=False)
        try:
            with pytest.raises((CollaborationConflict, CollaborationUnavailable)):
                await app.inspect_collaboration_request(admission.expected, context=context)
        finally:
            async with native_stores[0]._transaction(
                initialized.owner.application_scope, write=True
            ) as tx:
                await tx.put("requests", key, retained, insert=False)
    for changed in (
        occurrence.model_copy(update={"kind": "started"}),
        occurrence.model_copy(update={"sequence": 2}),
        occurrence.model_copy(update={"expected_revision": prior.revision + 1}),
    ):
        with pytest.raises(CollaborationConflict):
            await app.record_producer_progress(registration, changed, context=context)
    # Even an identical serialized owner command is not a raw public grant.
    with pytest.raises(CollaborationContractError):
        await app.record_collaboration_progress(first.command, context=context)
    async with native_stores[0]._transaction(initialized.owner.application_scope, write=True) as tx:
        with pytest.raises(CollaborationConflict):
            await progress_in_transaction(
                native_stores[0], tx, initialized, first.command, redactor=app._secret_redactor
            )
    destination = registration.destinations[0].operation
    await app.export_producer_output(registration, destination, context=disclosure)
    elected = await app.publish_producer_outcome(
        registration, destination=destination, context=disclosure
    )
    assert elected.command.terminal_frontier == 1
    assert await app.record_producer_progress(registration, occurrence, context=context) == first
    with pytest.raises(CollaborationConflict):
        await app.record_producer_progress(
            registration,
            occurrence.model_copy(
                update={
                    "operation": initialized.operation("late-progress"),
                    "sequence": 2,
                    "expected_revision": elected.revision,
                }
            ),
            context=context,
        )
    after = await app.inspect_collaboration_request(admission.expected, context=context)
    assert after.outcome == elected and len(after.progress) == 1
    assert after.progress[0].operation == first.command.operation
    assert after.progress[0].revision == first.revision
    assert len(provider.requests) == 1
