"""Planner admission is inert until exact producer responsibility is attached."""

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._contracts import CollaborationConflict, ExactNotFound
from cayu.collaboration.requests import RequestControl


@pytest.mark.anyio
async def test_planned_admission_closed_before_attachment_cannot_launch(native_stores):
    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        proposal,
        execution,
    ) = await output_scenario(native_stores, planned=True)
    snapshot = await app.inspect_collaboration_request(
        admission.expected, context=resolver.recipient.context
    )
    control = RequestControl(
        operation=initialized.operation("close-before-producer-attachment"),
        expected=admission.expected,
        expected_revision=snapshot.revision,
        kind="cancel",
    )
    closed = await app.control_collaboration_request(control, context=resolver.sender.context)
    assert closed.state == "cancelled"
    assert (
        await app.control_collaboration_request(control, context=resolver.sender.context) == closed
    )
    with pytest.raises(CollaborationConflict):
        await app.register_producer_output(proposal, execution, context=resolver.recipient.context)
    assert isinstance(
        await app.lookup_producer_registration(proposal, context=CONTEXT), ExactNotFound
    )
    assert (
        await app.inspect_participant(admission.prepared.recipient, context=CONTEXT)
    ).outstanding_obligations == 0
    assert await native_stores[1].load(session.id) == session
    assert provider.requests == []
