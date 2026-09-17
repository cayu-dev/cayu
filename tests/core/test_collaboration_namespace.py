"""Namespace elections through real transactional owners and identity admission."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_participant_identity import (
    CONTEXT,
    app,
    configuration,
    create,
    registration,
)

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    InitiatorBinding,
)
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.lifecycle import (
    CollaborationNamespaceRetired,
    LifecycleCommand,
    LifecycleIntent,
    NamespacePrune,
    NamespaceRetire,
    NamespaceRotate,
    NamespaceSeal,
)
from cayu.collaboration.participants import CollaborationCapacityExceeded, ParticipantCreate
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores
REDACTOR = SecretRedactor()


def command(initialized, request, *, principal="operator"):
    return LifecycleCommand(
        operation=request.operation,
        kind=request.kind,
        source=initialized.owner,
        destination=initialized.owner,
        initiator=InitiatorBinding(
            issuer=initialized.owner,
            principal=principal,
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        ),
        intent=LifecycleIntent(request=request, limits=initialized.binding.limits),
    )


async def rotate(store, initialized, key="rotate"):
    before = await store.inspect_namespace(initialized, redactor=REDACTOR)
    expected = command(
        initialized,
        NamespaceRotate(
            operation=before.current.reference.operation(key),
            namespace=before.current.reference,
            expected_revision=before.current.revision,
        ),
    )
    return expected, await store.apply_lifecycle(initialized, expected, redactor=REDACTOR)


async def test_rotation_preserves_bootstrap_and_exact_old_receipts(stores):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    request, created = await create(application, initialized)
    expected, rotated = await rotate(store, initialized)
    assert rotated.namespace.state == "sealed"
    assert rotated.successor.reference.generation == 2
    assert await application.initialize_collaboration() == initialized
    assert await application.create_participant(request, context=CONTEXT) == created
    assert await store.apply_lifecycle(initialized, expected, redactor=REDACTOR) == rotated
    with pytest.raises(CollaborationConflict):
        await create(application, initialized, "old-key")
    second = await application.create_participant(
        ParticipantCreate(
            operation=rotated.successor.reference.operation("new-key"),
            configuration=configuration(),
        ),
        context=CONTEXT,
    )
    assert second.event.sequence == rotated.event.sequence + 1
    recovered = stores()
    assert await recovered.initialize(reg.bootstrap, redactor=REDACTOR) == initialized
    assert (
        await recovered.inspect_namespace(initialized, redactor=REDACTOR)
    ).current == rotated.successor
    assert await recovered.apply_lifecycle(initialized, expected, redactor=REDACTOR) == rotated


async def test_sealed_namespace_can_rotate_but_not_admit(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    current = (await store.inspect_namespace(initialized, redactor=REDACTOR)).current
    seal = command(
        initialized,
        NamespaceSeal(
            operation=current.reference.operation("seal"),
            namespace=current.reference,
            expected_revision=current.revision,
        ),
    )
    sealed = await store.apply_lifecycle(initialized, seal, redactor=REDACTOR)
    with pytest.raises(CollaborationConflict):
        await create(application, initialized)
    _, rotated = await rotate(store, initialized)
    assert rotated.namespace.revision == sealed.namespace.revision + 1
    assert await store.apply_lifecycle(initialized, seal, redactor=REDACTOR) == sealed


async def test_identity_and_lifecycle_share_one_key_registry(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    await create(application, initialized, "shared")
    with pytest.raises(CollaborationConflict):
        await rotate(store, initialized, "shared")
    _, rotated = await rotate(store, initialized)
    with pytest.raises(CollaborationConflict):
        await create(application, initialized, "rotate")
    assert rotated.event.sequence == 3


async def test_retirement_is_contiguous_and_receipts_remain_historical(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    original_request, original = await create(application, initialized)
    _, first = await rotate(store, initialized)
    _, second = await rotate(store, initialized)
    later_first = command(
        initialized,
        NamespaceRetire(
            operation=second.successor.reference.operation("retire2"),
            namespace=second.namespace.reference,
            expected_revision=second.namespace.revision,
            expected_retired_through=1,
        ),
    )
    with pytest.raises(CollaborationConflict):
        await store.apply_lifecycle(initialized, later_first, redactor=REDACTOR)
    retirement = command(
        initialized,
        NamespaceRetire(
            operation=second.successor.reference.operation("retire1"),
            namespace=first.namespace.reference,
            expected_revision=first.namespace.revision,
            expected_retired_through=0,
        ),
    )
    receipt = await store.apply_lifecycle(initialized, retirement, redactor=REDACTOR)
    assert receipt.namespace.state == "retired"
    await store.apply_lifecycle(initialized, later_first, redactor=REDACTOR)
    inspection = await store.inspect_namespace(initialized, redactor=REDACTOR)
    assert inspection.retired_through == 2 and inspection.current.reference.generation == 3
    assert await store.apply_lifecycle(initialized, retirement, redactor=REDACTOR) == receipt
    assert await application.create_participant(original_request, context=CONTEXT) == original
    with pytest.raises(CollaborationNamespaceRetired):
        # The low-level owner preserves typed retirement separately from lookup.
        await store.apply(
            initialized,
            original.expected.model_copy(
                update={
                    "operation": initialized.operation("fresh"),
                    "intent": original.expected.intent.model_copy(
                        update={
                            "request": original_request.model_copy(
                                update={
                                    "operation": initialized.operation("fresh"),
                                }
                            )
                        }
                    ),
                }
            ),
            redactor=REDACTOR,
        )


async def test_concurrent_rotation_elects_once(stores):
    first, second = stores(), stores()
    initialized = await first.initialize(registration().bootstrap, redactor=REDACTOR)
    current = (await first.inspect_namespace(initialized, redactor=REDACTOR)).current
    requests = [
        command(
            initialized,
            NamespaceRotate(
                operation=current.reference.operation(key),
                namespace=current.reference,
                expected_revision=current.revision,
            ),
        )
        for key in ("one", "two")
    ]
    results = await asyncio.gather(
        *(
            store.apply_lifecycle(initialized, expected, redactor=REDACTOR)
            for store, expected in zip((first, second), requests, strict=True)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(value, CollaborationConflict) for value in results) == 1
    assert (
        await first.inspect_namespace(initialized, redactor=REDACTOR)
    ).current.reference.generation == 2
    events = await first.scan(
        initialized, table="events", after=0, limit=64, allowed=None, redactor=REDACTOR
    )
    assert len(events) == 2


async def test_rotation_capacity_rejection_is_atomic(stores):
    store = stores()
    reg = registration()
    binding = reg.bootstrap.model_copy(
        update={"limits": reg.bootstrap.limits.model_copy(update={"generations": 1})}
    )
    initialized = await store.initialize(binding, redactor=REDACTOR)
    before = await store.inspect_namespace(initialized, redactor=REDACTOR)
    with pytest.raises(CollaborationCapacityExceeded):
        await rotate(store, initialized)
    assert await store.inspect_namespace(initialized, redactor=REDACTOR) == before
    events = await store.scan(
        initialized, table="events", after=0, limit=64, allowed=None, redactor=REDACTOR
    )
    assert len(events) == 1


async def test_sealed_current_namespace_reclaims_generation_capacity_after_reopen(stores):
    from tests.core.test_collaboration_permits import Receiver, permit

    reg = registration()
    reg = registration(limits=reg.bootstrap.limits.model_copy(update={"generations": 2}))
    store = stores()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    pending = permit(initialized, ref)
    await store._register_permit(initialized, pending, redactor=REDACTOR)
    first = (await application.inspect_collaboration_namespace(context=CONTEXT)).current
    rotated = await application.rotate_collaboration_namespace(
        NamespaceRotate(
            operation=first.reference.operation("rotate"),
            namespace=first.reference,
            expected_revision=first.revision,
        ),
        context=CONTEXT,
    )
    current = rotated.successor
    assert current is not None
    seal = NamespaceSeal(
        operation=current.reference.operation("seal"),
        namespace=current.reference,
        expected_revision=current.revision,
    )
    sealed = await application.seal_collaboration_namespace(seal, context=CONTEXT)
    # Persistent backends reconstruct from a physically closed connection.
    if not isinstance(store, identity_tests.InMemoryCollaborationStore):
        await store.close()
    store = stores()
    application = app(store, reg)
    assert await application.initialize_collaboration() == initialized
    assert await application.seal_collaboration_namespace(seal, context=CONTEXT) == sealed
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    assert before.retained_generations == 2 and before.current.state == "sealed"
    next_rotation = NamespaceRotate(
        operation=current.reference.operation("next-rotation"),
        namespace=current.reference,
        expected_revision=before.current.revision,
    )
    with pytest.raises(CollaborationCapacityExceeded):
        await application.rotate_collaboration_namespace(next_rotation, context=CONTEXT)
    with pytest.raises(CollaborationConflict):
        await application.create_participant(
            ParticipantCreate(
                operation=current.reference.operation("blocked"), configuration=configuration()
            ),
            context=CONTEXT,
        )
    blocked_permit = permit(initialized, ref, key="blocked-permit")
    request = blocked_permit.intent.request.model_copy(
        update={
            "operation": current.reference.operation("blocked-permit"),
            "settlement_operation": current.reference.operation("blocked-settlement"),
        }
    )
    blocked_permit = blocked_permit.model_copy(
        update={
            "operation": request.operation,
            "intent": blocked_permit.intent.model_copy(update={"request": request}),
        }
    )
    with pytest.raises(CollaborationConflict):
        await store._register_permit(initialized, blocked_permit, redactor=REDACTOR)
    retirement = NamespaceRetire(
        operation=current.reference.operation("retire"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    with pytest.raises(CollaborationConflict):
        await application.retire_collaboration_namespace(retirement, context=CONTEXT)
    assert await application.inspect_collaboration_namespace(context=CONTEXT) == before
    await store._settle_permit(initialized, pending, reader=Receiver(pending), redactor=REDACTOR)
    retired = await application.retire_collaboration_namespace(retirement, context=CONTEXT)
    pruning = NamespacePrune(
        operation=current.reference.operation("prune"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before.retention_revision,
    )
    pruned = await application.prune_collaboration_namespace(pruning, context=CONTEXT)
    assert pruned.complete
    after = await application.inspect_collaboration_namespace(context=CONTEXT)
    assert after.retained_generations == 1 and after.current.state == "sealed"
    if not isinstance(store, identity_tests.InMemoryCollaborationStore):
        await store.close()
    store = stores()
    application = app(store, reg)
    await application.initialize_collaboration()
    assert await application.retire_collaboration_namespace(retirement, context=CONTEXT) == retired
    assert await application.prune_collaboration_namespace(pruning, context=CONTEXT) == pruned
    successor = await application.rotate_collaboration_namespace(next_rotation, context=CONTEXT)
    assert successor.successor.reference.generation == 3
    assert successor.successor.state == "open"
    # Historical replay remains valid, but fresh maintenance needs current authority.
    assert await application.prune_collaboration_namespace(pruning, context=CONTEXT) == pruned
    for generation in (2, 4):
        stale = pruning.model_copy(
            update={
                "operation": pruning.operation.model_copy(
                    update={
                        "generation": generation,
                        "caller_key": "wrong-control",
                    }
                ),
                "expected_retention_revision": pruned.retention_revision,
            }
        )
        with pytest.raises(CollaborationConflict):
            await application.prune_collaboration_namespace(stale, context=CONTEXT)
    await application.create_participant(
        ParticipantCreate(
            operation=successor.successor.reference.operation("new"), configuration=configuration()
        ),
        context=CONTEXT,
    )


@pytest.mark.parametrize("kind", ["seal", "rotate", "retire", "prune"])
@pytest.mark.parametrize("loss", ["acknowledgement", "cancellation"])
async def test_namespace_commit_survives_lost_observation(stores, monkeypatch, kind, loss):
    store = stores()
    reg = registration()
    initialized = await store.initialize(reg.bootstrap, redactor=REDACTOR)
    current = (await store.inspect_namespace(initialized, redactor=REDACTOR)).current
    if kind in ("retire", "prune"):
        _, prior = await rotate(store, initialized)
        request = NamespaceRetire(
            operation=prior.successor.reference.operation("control"),
            namespace=prior.namespace.reference,
            expected_revision=prior.namespace.revision,
            expected_retired_through=0,
        )
        if kind == "prune":
            await store.apply_lifecycle(
                initialized, command(initialized, request), redactor=REDACTOR
            )
            request = NamespacePrune(
                operation=prior.successor.reference.operation("prune"),
                namespace=prior.namespace.reference,
                expected_retention_revision=(
                    await store.inspect_namespace(initialized, redactor=REDACTOR)
                ).retention_revision,
            )
    else:
        schema = NamespaceSeal if kind == "seal" else NamespaceRotate
        request = schema(
            operation=current.reference.operation("control"),
            namespace=current.reference,
            expected_revision=current.revision,
        )
    expected = command(initialized, request)
    original = store._transaction
    committed, release = asyncio.Event(), asyncio.Event()
    injected = False

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal injected
        async with original(scope, write=write) as tx:
            yield tx
        if write and not injected:
            injected = True
            committed.set()
            await release.wait()
            if loss == "acknowledgement":
                raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(store, "_transaction", transaction)
    caller = asyncio.create_task(store.apply_lifecycle(initialized, expected, redactor=REDACTOR))
    retry = None
    try:
        await asyncio.wait_for(committed.wait(), 5)
        if loss == "cancellation":
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
            assert len(store._owners.pending) == 1
            retry = asyncio.create_task(
                store.apply_lifecycle(initialized, expected, redactor=REDACTOR)
            )
            await asyncio.sleep(0)
            assert not retry.done()
        release.set()
        if loss == "acknowledgement":
            with pytest.raises(ConnectionError):
                await caller
            receipt = await store.apply_lifecycle(initialized, expected, redactor=REDACTOR)
        else:
            receipt = await retry
        recovered = stores()
        assert await recovered.initialize(reg.bootstrap, redactor=REDACTOR) == initialized
        assert await recovered.apply_lifecycle(initialized, expected, redactor=REDACTOR) == receipt
        events = await recovered.scan(
            initialized,
            table="events",
            after=0,
            limit=64,
            allowed=None,
            redactor=REDACTOR,
        )
        assert sum(event.operation == expected.operation for event in events) == 1
        changed = expected.model_copy(
            update={"initiator": expected.initiator.model_copy(update={"principal": "different"})}
        )
        with pytest.raises(CollaborationConflict):
            await recovered.apply_lifecycle(initialized, changed, redactor=REDACTOR)
    finally:
        release.set()
        await asyncio.gather(caller, *(() if retry is None else (retry,)), return_exceptions=True)


@pytest.mark.parametrize("secret", ["sealed", "namespace_rotated", "collaboration.lifecycle"])
async def test_public_namespace_journey_and_authorized_replay(stores, secret):
    store = stores()
    policy = identity_tests.Policy()
    reg = registration(policy=policy)
    application = app(store, reg, secret_redactor=SecretRedactor(secret))
    initialized = await application.initialize_collaboration()
    initial = await application.inspect_collaboration_namespace(context=CONTEXT)
    request = NamespaceRotate(
        operation=initial.current.reference.operation("rotate"),
        namespace=initial.current.reference,
        expected_revision=1,
    )
    rotated = await application.rotate_collaboration_namespace(request, context=CONTEXT)
    found = await application.lookup_collaboration_lifecycle_operation(
        rotated.expected, context=CONTEXT
    )
    assert isinstance(found, ExactMatch) and found.receipt == rotated
    changed = rotated.expected.model_copy(
        update={
            "initiator": rotated.expected.initiator.model_copy(update={"principal": "another"}),
        }
    )
    assert isinstance(
        await application.lookup_collaboration_lifecycle_operation(changed, context=CONTEXT),
        ExactConflict,
    )
    missing_request = request.model_copy(
        update={"operation": initial.current.reference.operation("missing")}
    )
    assert isinstance(
        await application.lookup_collaboration_lifecycle_operation(
            command(initialized, missing_request), context=CONTEXT
        ),
        ExactNotFound,
    )
    policy.denied.add("namespace_rotate")
    assert await application.rotate_collaboration_namespace(request, context=CONTEXT) == rotated
    participant = await application.create_participant(
        ParticipantCreate(
            operation=rotated.successor.reference.operation("create"),
            configuration=configuration(),
        ),
        context=CONTEXT,
    )
    # A participant-scoped grant is insufficient to mutate or inspect namespace authority.
    policy.allowed = (participant.participants[0].reference,)
    with pytest.raises(CollaborationAccessDenied):
        await application.inspect_collaboration_namespace(context=CONTEXT)
    with pytest.raises(CollaborationAccessDenied):
        await application.rotate_collaboration_namespace(request, context=CONTEXT)
    policy.allowed = None
    retirement = NamespaceRetire(
        operation=rotated.successor.reference.operation("retire"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    retired = await application.retire_collaboration_namespace(retirement, context=CONTEXT)
    assert retired.namespace.state == "retired"
    assert await application.initialize_collaboration() == initialized
    with pytest.raises(CollaborationNamespaceRetired):
        await create(application, initialized, "fresh-after-retirement")
    page = await application.list_participant_events(context=CONTEXT)
    assert [event.type for event in page.events] == [
        "initialized",
        "namespace_rotated",
        "created",
        "namespace_retired",
    ]
