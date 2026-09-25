"""Native namespace reclamation removes plans before their request evidence."""

from contextlib import asynccontextmanager

import pytest
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT, app, stores
from tests.core.test_request_planning_public import complete_plan, scenario

from cayu.collaboration._contracts import ExactMatch, ExactUnavailable, ObjectRef
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import (
    RequestReceivingAuthorization,
    RequestReceivingOwner,
    RequestRegistration,
)
from cayu.collaboration.requests import RequestAdmissionCommand, RequestControl

__all__ = ["stores"]
pytestmark = pytest.mark.anyio


async def test_root_control_and_plan_cleanup_rollback_together(stores, monkeypatch):
    store = stores()
    application, resolver, command, _, _ = await scenario(store, "defer")
    planned = await complete_plan(application, command, resolver.recipient.context)
    initialized = await application.initialize_collaboration()
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.recipient.context
    )
    control = RequestControl(
        operation=initialized.operation("atomic-root-plan-control"),
        expected=command.expected,
        expected_revision=2,
        kind="cancel",
    )
    original = store._transaction

    @asynccontextmanager
    async def fail_after_plan_event(scope, *, write):
        async with original(scope, write=write) as tx:
            put = tx.put

            async def failing_put(table, key, value, *, insert):
                await put(table, key, value, insert=insert)
                if table == "request_plan_events":
                    raise OSError("plan control event acknowledgement failed")

            if write:
                tx.put = failing_put
            yield tx

    with monkeypatch.context() as patch:
        patch.setattr(store, "_transaction", fail_after_plan_event)
        with pytest.raises(CollaborationUnavailable):
            await application.control_collaboration_request(
                control, context=resolver.recipient.context
            )
    assert (
        await application.inspect_collaboration_request(
            command.expected, context=resolver.recipient.context
        )
        == before
    )
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch) and found.receipt == planned
    result = await application.control_collaboration_request(
        control, context=resolver.recipient.context
    )
    assert result.state == "cancelled"


async def test_local_defer_does_not_settle_preceding_registered_receiver_responsibility(stores):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    initialized = await application.initialize_collaboration()
    admission = RequestAdmissionCommand(
        operation=initialized.operation("external-owner-defer"),
        expected=command.expected,
        expected_revision=1,
        expected_input_revision=command.expected_input_revision,
        expected_input_sha256=command.expected_input_sha256,
        generation=1,
        decision="defer",
        evidence=(),
        initiator=command.initiator,
    )

    class Receiver(RequestReceivingOwner):
        @property
        def ref(self):
            return ObjectRef(
                owner=initialized.owner,
                kind="receiver",
                object_id="independent-receiver",
                incarnation="one",
                revision=1,
            )

        @asynccontextmanager
        async def acquire(self, expected, *, context):
            assert expected == admission
            yield RequestReceivingAuthorization(
                receiver=self.ref,
                command=expected,
                expires_at_ms=4102444800000,
            )

    admitting = app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(
            mandates=resolver, max_ttl_ms=60000, receiving_owner=Receiver()
        ),
    )
    await admitting.initialize_collaboration()
    await admitting.admit_collaboration_request(admission, context=resolver.recipient.context)
    command = command.model_copy(update={"expected_revision": 2, "admission_generation": 2})
    planned = await complete_plan(application, command, resolver.recipient.context)
    with pytest.raises(CollaborationUnavailable):
        await application.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("must-not-settle-foreign-owner"),
                expected=command.expected,
                expected_revision=3,
                kind="cancel",
            ),
            context=resolver.recipient.context,
        )
    found = await application.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch) and found.receipt == planned
    assert provider.requests == []


@pytest.mark.parametrize("kind", ["cancel", "expire"])
async def test_public_root_control_settles_only_local_deferral_after_reconstruction(
    stores, monkeypatch, kind
):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "defer")
    planned = await complete_plan(application, command, resolver.recipient.context)
    reopened_store = stores()
    reopened = app(
        reopened_store,
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=60000),
    )
    initialized = await reopened.initialize_collaboration()
    if kind == "expire":
        original = reopened_store._transaction

        @asynccontextmanager
        async def expired_owner(scope, *, write):
            async with original(scope, write=write) as tx:

                async def now_ms():
                    return command.expected.intent.selection.expires_at_ms

                tx.now_ms = now_ms
                yield tx

        monkeypatch.setattr(reopened_store, "_transaction", expired_owner)
    control = RequestControl(
        operation=initialized.operation("cancel-local-plan"),
        expected=command.expected,
        expected_revision=2,
        kind=kind,
    )
    terminal = await reopened.control_collaboration_request(
        control, context=resolver.recipient.context
    )
    assert terminal.state == ("expired" if kind == "expire" else "cancelled")
    found = await reopened.lookup_collaboration_plan(command, context=resolver.recipient.context)
    assert isinstance(found, ExactMatch)
    assert found.receipt.state == terminal.state
    assert found.receipt.revision == planned.revision + 1
    assert found.receipt.pending_stages == found.receipt.reserved_events == 0
    assert found.receipt.reserved_bytes == 0
    assert (
        await reopened.control_collaboration_request(control, context=resolver.recipient.context)
        == terminal
    )
    async with reopened_store._transaction(
        initialized.binding.application_scope, write=False
    ) as tx:
        assert not await tx.scan_pending_request_plans(after=None, limit=1)
    assert provider.requests == []


async def test_public_declined_plan_prunes_with_original_request_in_small_batches(stores):
    store = stores()
    application, resolver, command, _, provider = await scenario(store, "decline")
    await complete_plan(application, command, resolver.recipient.context)
    initialized = await application.initialize_collaboration()
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-planned"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    for index in range(20):
        before = await application.inspect_collaboration_namespace(context=CONTEXT)
        pruning = NamespacePrune(
            operation=rotated.successor.reference.operation(f"prune-planning-{index}"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=before.retention_revision,
            max_records=2,
        )
        result = await application.prune_collaboration_namespace(pruning, context=CONTEXT)
        assert result.removed_records <= 2
        assert await application.prune_collaboration_namespace(pruning, context=CONTEXT) == result
        if result.complete:
            break
    else:
        pytest.fail("Planning blocked bounded reclamation of a settled namespace.")
    assert isinstance(
        await application.lookup_collaboration_plan(command, context=resolver.recipient.context),
        ExactUnavailable,
    )
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert not await tx.scan_request_plans(command.expected.intent.selection.reference, limit=1)
        assert not await tx.scan_pending_request_plans(after=None, limit=1)
    assert provider.requests == []
