"""Public terminal cleanup after a real assistant-export and peer-delivery flow."""

import pytest
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT, app

from cayu import ClarificationExpiryRequest, ClarificationQuestionRecovery
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.vaults.redaction import SecretRedactor


async def prune_closed_question(
    application, factory, initialized, command, actor, mandates, registration
):
    instances = []

    def reopen():
        store = factory()
        instances.append(store)
        return store

    try:
        await _prune_closed_question(
            application, reopen, initialized, command, actor, mandates, registration
        )
    finally:
        for store in instances:
            # Memory factories deliberately share one owner with the surrounding
            # runtime fixture; closing it here would terminate unrelated work.
            if not isinstance(store, InMemoryCollaborationStore):
                await store.close()


async def _prune_closed_question(
    application, factory, initialized, command, actor, mandates, registration
):
    expected = command.expected
    snapshot = await application.inspect_collaboration_request(expected, context=actor.context)
    assert snapshot.state == "answered"
    assert snapshot.clarification.input_revision == 1
    store = factory()
    redactor = SecretRedactor()
    expiry = ClarificationExpiryRequest(
        operation=initialized.operation("expiry-after-answer"),
        recovery=ClarificationQuestionRecovery(
            operation=command.question.operation,
            question_sha256=clarification_commitment(command.question, redactor),
            request_sha256=clarification_commitment(command.expected, redactor),
        ),
    )
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        before = await store._anchor(tx, initialized, redactor)
    with pytest.raises(CollaborationConflict):
        await application.expire_clarification_question(expiry, context=CONTEXT)
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, redactor) == before
        assert await tx.get("operations", operation_key(expiry.operation)) is None
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-history"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    partial_seen = False
    for index in range(40):
        state = await application.inspect_collaboration_namespace(context=CONTEXT)
        batch = NamespacePrune(
            operation=rotated.successor.reference.operation(f"prune-history-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=state.retention_revision,
            max_records=3,
        )
        receipt = await application.prune_collaboration_namespace(batch, context=CONTEXT)
        assert receipt.removed_records <= 3
        assert await application.prune_collaboration_namespace(batch, context=CONTEXT) == receipt
        async with store._transaction(initialized.owner.application_scope, write=False) as tx:
            progress = await tx.get("request_pruning", operation_key(expected.operation))
        if progress is not None:
            partial_seen = True
            with pytest.raises(CollaborationUnavailable):
                await application.inspect_collaboration_request(expected, context=actor.context)
            # Reconstruct public ownership after each partial batch. No original
            # native transaction or in-process cursor is carried to the new app.
            application = app(
                factory(),
                registration,
                collaboration_requests=RequestRegistration(mandates=mandates, max_ttl_ms=300000),
            )
            assert await application.initialize_collaboration() == initialized
            assert (
                await application.prune_collaboration_namespace(batch, context=CONTEXT) == receipt
            )
        if receipt.complete:
            break
    else:
        pytest.fail("Terminal clarification history was not reclaimed in bounded batches.")
    assert partial_seen
    async with store._transaction(initialized.owner.application_scope, write=False) as tx:
        assert await tx.get("requests", operation_key(expected.operation)) is None
        assert await tx.get("request_pruning", operation_key(expected.operation)) is None
        assert (
            await tx.get("clarification_questions", operation_key(command.question.operation))
            is None
        )
        assert (
            await tx.get("clarification_lineages", operation_key(command.question.lineage)) is None
        )
        assert not await tx.scan_clarification_request_handoffs(
            expected.intent.selection.reference, family="clarification_deliveries", limit=1
        )
        assert not await tx.scan_clarification_request_handoffs(
            expected.intent.selection.reference, family="clarification_services", limit=1
        )
        reference = expected.intent.selection.reference
        assert (
            await tx.get("clarification_inputs", (reference.request_id, reference.incarnation, 1))
            is None
        )
