"""Registered occurrence ceilings and native milestones fail closed before dispatch."""

import pytest
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu import ProducerProgressOccurrence
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
async def test_prepared_progress_requires_native_evidence_and_enforces_occurrence_bound(
    native_stores, monkeypatch
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
    ) = await output_scenario(native_stores)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    command = command.model_copy(
        update={"limits": command.limits.model_copy(update={"progress_occurrences": 1})}
    )
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    registered_session = await native_stores[1].load(session.id)
    resolution = resolver.recipient.resolution
    actions = tuple(dict.fromkeys((*resolution.principal.actions, "publish")))
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
    context = resolver.recipient.context
    prior = await app.inspect_collaboration_request(admission.expected, context=context)
    occurrence = ProducerProgressOccurrence(
        operation=initialized.operation("bounded-native-progress"),
        expected_revision=prior.revision,
        sequence=1,
        kind="started",
    )
    for kind in ("started", "producing", "published"):
        with pytest.raises(CollaborationUnavailable):
            await app.record_producer_progress(
                command, occurrence.model_copy(update={"kind": kind}), context=context
            )
    occurrence = occurrence.model_copy(update={"kind": "prepared"})
    with monkeypatch.context() as patch:
        patch.setattr(app.session_store, "_supports_producer_attachment_protocol", lambda: False)
        with pytest.raises(CollaborationUnavailable):
            await app.record_producer_progress(command, occurrence, context=context)
    first = await app.record_producer_progress(command, occurrence, context=context)
    assert (
        first.command.evidence.interaction_id is None and first.command.evidence.run_epoch is None
    )
    assert await app.record_producer_progress(command, occurrence, context=context) == first
    with pytest.raises(CollaborationConflict):
        await app.record_producer_progress(
            command,
            occurrence.model_copy(
                update={
                    "operation": initialized.operation("over-limit-progress"),
                    "expected_revision": first.revision,
                    "sequence": 2,
                }
            ),
            context=context,
        )
    after = await app.inspect_collaboration_request(admission.expected, context=context)
    assert len(after.progress) == 1 and after.state == "open"
    assert after.progress[0].operation == first.command.operation
    assert after.progress[0].revision == first.revision
    assert await native_stores[1].load(session.id) == registered_session
    assert not provider.requests
