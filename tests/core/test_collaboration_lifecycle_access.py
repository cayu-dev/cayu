"""Public lifecycle authority, including unsupported and unelected namespaces."""

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_participant_identity import CONTEXT, app, registration

from cayu.collaboration._capabilities import (
    CapabilityDescriptor,
    CollaborationCapabilityUnavailable,
)
from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
from cayu.collaboration.base import IDENTITY_FAMILY, LIFECYCLE_FAMILY
from cayu.collaboration.lifecycle import NamespaceRotate

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


@pytest.mark.parametrize("difference", ["generation", "namespace_incarnation", "owner"])
async def test_unelected_namespace_cannot_be_inspected_or_provisioned(stores, difference):
    store = stores()
    application = app(store, registration())
    await application.initialize_collaboration()
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    namespace = before.current.reference
    changes = {
        "generation": namespace.generation + 1,
        "namespace_incarnation": "unelected",
        "owner": namespace.owner.model_copy(update={"incarnation": "unelected"}),
    }
    unknown = namespace.model_copy(update={difference: changes[difference]})
    events = await application.list_participant_events(context=CONTEXT)
    with pytest.raises(CollaborationConflict):
        await application.inspect_collaboration_retirement(unknown, context=CONTEXT)
    with pytest.raises((CollaborationConflict, CollaborationContractError)) as caught:
        await application.rotate_collaboration_namespace(
            NamespaceRotate(
                operation=unknown.operation("unknown"),
                namespace=unknown,
                expected_revision=1,
            ),
            context=CONTEXT,
        )
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert await application.inspect_collaboration_namespace(context=CONTEXT) == before
    assert await application.list_participant_events(context=CONTEXT) == events


async def test_lifecycle_capability_rejects_new_mutation_but_preserves_exact_readback(
    stores, monkeypatch
):
    store = stores()
    application = app(store, registration())
    await application.initialize_collaboration()
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    request = NamespaceRotate(
        operation=before.current.reference.operation("rotate"),
        namespace=before.current.reference,
        expected_revision=1,
    )
    receipt = await application.rotate_collaboration_namespace(request, context=CONTEXT)
    monkeypatch.setattr(
        store,
        "capabilities",
        lambda owner: CapabilityDescriptor(
            owner=owner,
            mutations=(IDENTITY_FAMILY,),
            readbacks=(IDENTITY_FAMILY, LIFECYCLE_FAMILY),
        ),
    )
    assert await application.rotate_collaboration_namespace(request, context=CONTEXT) == receipt
    events = await application.list_participant_events(context=CONTEXT)
    current = await application.inspect_collaboration_namespace(context=CONTEXT)
    with pytest.raises(CollaborationCapabilityUnavailable):
        await application.rotate_collaboration_namespace(
            NamespaceRotate(
                operation=current.current.reference.operation("another"),
                namespace=current.current.reference,
                expected_revision=1,
            ),
            context=CONTEXT,
        )
    assert await application.inspect_collaboration_namespace(context=CONTEXT) == current
    assert await application.list_participant_events(context=CONTEXT) == events
