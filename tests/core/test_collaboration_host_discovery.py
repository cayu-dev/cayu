"""Host scans use genuine source-owner APIs, not process-local work registries."""

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT

from cayu.collaboration._host_discovery import discover_host_source
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.requests import RequestControl

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


@pytest.mark.parametrize("source", ["plans", "questions", "deliveries", "services", "waits"])
async def test_host_empty_native_discovery_lanes(stores, source):
    application, resolver, _values = await public_setup(stores())
    page = await discover_host_source(
        application,
        source,
        context=resolver.context if source == "plans" else CONTEXT,
        limit=1,
        max_bytes=65536,
    )
    assert page.source == source
    assert page.items == ()
    assert page.reached_scan_end


async def test_host_discovers_accepted_work_without_dispatch_and_restarts(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    page = await discover_host_source(
        application, "requests", context=resolver.context, limit=1, max_bytes=65536
    )
    assert len(page.items) == 1
    assert page.items[0].receipt == receipt
    assert page.items[0].admission == "undecided"
    assert page.retained_bytes > 0
    # A full page carries a continuation hint, not a claim of complete coverage.
    assert not page.reached_scan_end

    other = identity_tests.app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    replay = await discover_host_source(
        other, "requests", context=resolver.context, limit=1, max_bytes=65536
    )
    assert replay.items == page.items
    tail = await discover_host_source(
        other,
        "requests",
        context=resolver.context,
        cursor=page.next_cursor,
        limit=1,
        max_bytes=65536,
    )
    assert tail.items == () and tail.reached_scan_end
    # The read has not claimed, admitted, or settled accepted work.
    assert (
        await application.inspect_collaboration_request(receipt.expected, context=resolver.context)
        == page.items[0]
    )


async def test_host_discovery_refuses_wrong_context_and_capacity_without_mutation(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    with pytest.raises(TypeError, match="mandate"):
        await discover_host_source(
            application, "requests", context=CONTEXT, limit=1, max_bytes=65536
        )
    with pytest.raises(ValueError, match="byte limit"):
        await discover_host_source(
            application, "requests", context=resolver.context, limit=1, max_bytes=1
        )
    page = await discover_host_source(
        application, "requests", context=resolver.context, limit=1, max_bytes=65536
    )
    assert page.items[0].receipt == receipt
    assert page.items[0].admission == "undecided"


async def test_host_maintenance_discovery_reauthenticates_and_reports_empty_source(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    participant = receipt.expected.intent.selection.recipient.reference
    page = await discover_host_source(
        application,
        "producers",
        context=CONTEXT,
        participant=participant,
        limit=1,
        max_bytes=65536,
    )
    assert not page.items  # accepted responsibility is not a producer attachment
    # Native producer discovery may skip unrelated pending permits, retaining
    # a cursor even for an empty page. Do not turn this into global quiescence.
    with pytest.raises(CollaborationAccessDenied):
        await discover_host_source(
            application,
            "producers",
            context=CollaborationAccessContext(principal="unauthorized"),
            participant=participant,
            limit=1,
            max_bytes=65536,
        )


async def test_host_new_sweep_observes_source_transition(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    before = await discover_host_source(
        application, "requests", context=resolver.context, limit=1, max_bytes=65536
    )
    assert before.items
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("host-cancel"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    after = await discover_host_source(
        application, "requests", context=resolver.context, limit=1, max_bytes=65536
    )
    assert not after.items and after.reached_scan_end
    assert before.items[0].state == "open"  # detached snapshot, not mutable owner state
