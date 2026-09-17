"""Retired namespace pruning through the public application boundary."""

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_collaboration_permits import Receiver, permit
from tests.core.test_participant_identity import CONTEXT, app, configuration, create, registration

from cayu.collaboration.lifecycle import (
    CollaborationHistoryUnavailable,
    NamespacePrune,
    NamespaceRetire,
    ParticipantLifecycleChange,
)
from cayu.collaboration.participants import (
    CollaborationCapacityExceeded,
    ParticipantConfigure,
    ParticipantCreate,
)
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores
REDACTOR = SecretRedactor()


async def test_public_pruning_is_bounded_replayable_and_preserves_current_participant(stores):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    before = await store.inspect_namespace(initialized, redactor=REDACTOR)
    retired = await application.inspect_collaboration_retirement(
        rotated.namespace.reference,
        context=CONTEXT,
    )
    assert retired.content == "retained"
    assert (
        await application.inspect_collaboration_retirement(
            rotated.successor.reference,
            context=CONTEXT,
        )
        is None
    )
    event_page = await application.list_participant_events(context=CONTEXT, limit=1)
    assert event_page.history_complete
    assert event_page.next_cursor is not None
    request = NamespacePrune(
        operation=rotated.successor.reference.operation("prune-first"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before.retention_revision,
        max_records=1,
    )
    first = await application.prune_collaboration_namespace(request, context=CONTEXT)
    assert first.removed_records == 1
    assert not first.complete
    partial = await application.inspect_collaboration_retirement(
        rotated.namespace.reference,
        context=CONTEXT,
    )
    assert partial.content == "partial"
    assert (
        await application.lookup_participant_operation(
            created.expected,
            context=CONTEXT,
        )
    ).status == "unavailable"
    with pytest.raises(CollaborationHistoryUnavailable):
        await application.list_participant_events(context=CONTEXT, cursor=event_page.next_cursor)
    retained = await application.list_participant_events(context=CONTEXT)
    assert not retained.history_complete
    assert retained.retention_revision == first.retention_revision
    assert await application.prune_collaboration_namespace(request, context=CONTEXT) == first
    second = await application.prune_collaboration_namespace(
        NamespacePrune(
            operation=rotated.successor.reference.operation("prune-second"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=first.retention_revision,
            max_records=1,
        ),
        context=CONTEXT,
    )
    assert second.removed_records == 1
    assert second.complete
    after = await store.inspect_namespace(initialized, redactor=REDACTOR)
    assert after.pruned_through == 1
    pruned = await application.inspect_collaboration_retirement(
        rotated.namespace.reference,
        context=CONTEXT,
    )
    assert pruned.content == "pruned"
    assert after.retention_revision == before.retention_revision + 2
    # Current participant authority outlives the retired creation receipt.
    current = await application.inspect_participant(
        created.participants[0].reference, context=CONTEXT
    )
    assert current.participant == created.participants[0]
    reopened = app(stores(), reg)
    await reopened.initialize_collaboration()
    assert await reopened.prune_collaboration_namespace(request, context=CONTEXT) == first


async def test_pruning_settled_permit_is_atomic_and_invalidates_inventory_cursor(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized, key="z-create")
    ref = created.participants[0].reference
    expected = permit(initialized, ref, key="a-permit")
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=REDACTOR)
    inventory = await application.list_participant_obligations(
        ref,
        context=CONTEXT,
        pending_only=False,
        limit=1,
    )
    assert inventory.next_cursor is not None
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    request = NamespacePrune(
        operation=rotated.successor.reference.operation("prune"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before.retention_revision,
        max_records=1,
    )
    with pytest.raises(CollaborationCapacityExceeded):
        await application.prune_collaboration_namespace(request, context=CONTEXT)
    assert await application.inspect_collaboration_namespace(context=CONTEXT) == before
    unchanged = await application.list_participant_obligations(
        ref, context=CONTEXT, pending_only=False
    )
    assert unchanged.obligations == inventory.obligations
    accepted = request.model_copy(update={"max_records": 2})
    receipt = await application.prune_collaboration_namespace(accepted, context=CONTEXT)
    assert receipt.removed_records == 2 and not receipt.complete
    assert await application.prune_collaboration_namespace(accepted, context=CONTEXT) == receipt
    with pytest.raises(CollaborationHistoryUnavailable):
        await application.list_participant_obligations(
            ref,
            context=CONTEXT,
            pending_only=False,
            cursor=inventory.next_cursor,
        )
    assert not (
        await application.list_participant_obligations(
            ref,
            context=CONTEXT,
            pending_only=False,
        )
    ).obligations
    assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"


async def retire_and_prune(application, rotated, floor):
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=floor,
        ),
        context=CONTEXT,
    )
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    return await application.prune_collaboration_namespace(
        NamespacePrune(
            operation=rotated.successor.reference.operation("prune"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=before.retention_revision,
        ),
        context=CONTEXT,
    )


async def test_later_receipt_pins_old_configuration_until_its_generation_is_pruned(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    _, first = await rotate(store, initialized)
    close = ParticipantLifecycleChange(
        operation=first.successor.reference.operation("disable"),
        participant=ref,
        expected_lifecycle_revision=1,
        state="disabled",
    )
    disabled = await application.change_participant_lifecycle(close, context=CONTEXT)
    await application.configure_participant(
        ParticipantConfigure(
            operation=first.successor.reference.operation("configure"),
            participant=ref,
            expected_configuration_revision=1,
            configuration=configuration(2),
        ),
        context=CONTEXT,
    )
    assert (await retire_and_prune(application, first, 0)).complete
    assert await application.change_participant_lifecycle(close, context=CONTEXT) == disabled
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await tx.get("configurations", (ref.participant_id, 1)) is not None
        assert await tx.get("lifecycle_history", (ref.participant_id, 1)) is None
    _, second = await rotate(store, initialized)
    assert (await retire_and_prune(application, second, 1)).complete
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await tx.get("configurations", (ref.participant_id, 1)) is None
        assert await tx.get("configurations", (ref.participant_id, 2)) is not None
        assert await tx.get("lifecycle_history", (ref.participant_id, 2)) is not None
    # An unreferenced current history stays until superseded, then releases capacity.
    updated = await application.configure_participant(
        ParticipantConfigure(
            operation=second.successor.reference.operation("configure"),
            participant=ref,
            expected_configuration_revision=2,
            configuration=configuration(),
        ),
        context=CONTEXT,
    )
    assert updated.participants[0].configuration_revision == 3
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await tx.get("configurations", (ref.participant_id, 2)) is None
        assert await tx.get("configurations", (ref.participant_id, 3)) is not None
    assert (
        await application.inspect_participant(ref, context=CONTEXT)
    ).participant == updated.participants[0]


async def test_full_optional_capacity_settles_then_reclaims_and_readmits(stores):
    store = stores()
    limits = registration().bootstrap.limits.model_copy(
        update={
            "operations": 8,
            "control_operations": 3,
            "events": 10,
            "control_events": 3,
            "generations": 2,
        }
    )
    application = app(store, registration(limits=limits))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    await create(application, initialized, "extra-one")
    await create(application, initialized, "extra-two")
    with pytest.raises(CollaborationCapacityExceeded):
        await create(application, initialized, "over-capacity")
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("disable"),
            participant=ref,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=REDACTOR)
    _, rotated = await rotate(store, initialized)
    pruned = await retire_and_prune(application, rotated, 0)
    assert pruned.complete
    new = await application.create_participant(
        ParticipantCreate(
            operation=rotated.successor.reference.operation("new-work"),
            configuration=configuration(),
        ),
        context=CONTEXT,
    )
    assert new.participants[0].reference != ref
    assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"
