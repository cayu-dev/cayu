"""Native session discovery is scoped to exact participants, not public IDs."""

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario

from cayu.collaboration._host_discovery import discover_host_source
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied

pytestmark = pytest.mark.anyio


async def test_host_session_inventory_uses_native_creation_and_current_access(native_stores):
    application, resolver, command, provider, session, _ = await prepared_scenario(native_stores)
    participant = command.prepared.recipient
    page = await discover_host_source(
        application, "sessions", participant=participant, context=CONTEXT, limit=1, max_bytes=65536
    )
    assert len(page.items) == 1 and page.next_cursor is not None
    item = page.items[0]
    assert (item.session_id, item.session_instance_id) == (session.id, session.instance_id)
    creation = await application.session_store.load_participant_session_creation_receipt(session.id)
    assert item.receipt_commitment == creation.receipt_commitment
    assert item.participant == participant
    tail = await discover_host_source(
        application,
        "sessions",
        participant=participant,
        context=CONTEXT,
        cursor=page.next_cursor,
        limit=1,
        max_bytes=65536,
    )
    assert tail.items == () and tail.reached_scan_end
    with pytest.raises(CollaborationAccessDenied):
        await discover_host_source(
            application,
            "sessions",
            participant=participant,
            context=CollaborationAccessContext(principal="foreign"),
            limit=1,
            max_bytes=65536,
        )
    with pytest.raises(ValueError, match="another participant"):
        await discover_host_source(
            application,
            "sessions",
            participant=resolver.sender.context.participant,
            context=CONTEXT,
            cursor=page.next_cursor,
            limit=1,
            max_bytes=65536,
        )
    assert not provider.requests
    assert (
        await application.inspect_collaboration_request(
            command.expected, context=resolver.recipient.context
        )
        is not None
    )
