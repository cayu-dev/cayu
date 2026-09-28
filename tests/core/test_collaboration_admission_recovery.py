"""A lost admission handle is reconstructed from its authenticated source owner."""

import pytest
from tests.core.test_participant_identity import app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario

from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.collaboration.access import CollaborationAccessDenied

pytestmark = pytest.mark.anyio


async def test_admission_recovery_after_commit_and_reopen_is_inert(native_stores):
    application, resolver, command, provider, session, initialized = await prepared_scenario(
        native_stores
    )
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    assert isinstance(
        await application.recover_collaboration_admission(
            before, context=resolver.recipient.context
        ),
        ExactNotFound,
    )
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    current = await application.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    other = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    try:
        assert isinstance(
            await other.recover_collaboration_admission(before, context=resolver.recipient.context),
            ExactConflict,
        )
        recovered = await other.recover_collaboration_admission(
            current, context=resolver.recipient.context
        )
        assert isinstance(recovered, ExactMatch) and recovered.receipt == receipt
        assert not provider.requests
        assert await other.session_store.load(session.id) is None
        assert current.producer_operation is None
        assert (
            await application.inspect_collaboration_request(
                command.expected, context=resolver.recipient.context
            )
            == current
        )
        with pytest.raises(CollaborationAccessDenied):
            await other.recover_collaboration_admission(
                current,
                context=resolver.recipient.context.model_copy(update={"principal": "foreign"}),
            )
        # The complete durable admission must still have its authenticated event;
        # a command-shaped row and request index alone are not positive evidence.
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=True
        ) as tx:
            await tx.delete("request_events", (receipt.event.sequence,))
        assert isinstance(
            await other.recover_collaboration_admission(
                current, context=resolver.recipient.context
            ),
            ExactUnavailable,
        )
        assert not provider.requests
    finally:
        await other.drain_collaboration_requests()
        await application.drain_collaboration_requests()
