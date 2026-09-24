"""Public bounded retirement batches preserve exact replay across reconstruction."""

import pytest
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_participant_identity import stores as stores

from cayu.collaboration._request_store import operation_key
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestControl, RequestObservation


@pytest.mark.anyio
@pytest.mark.parametrize("observations,batch", [(0, 1), (32, 32)])
async def test_public_request_pruning_resumes_exact_frontier(
    stores, observations, batch, monkeypatch
):
    store = stores()
    # This test measures the durable batch frontier and rollback, not the
    # foreground observation deadline. A 32-record SQL batch may outlive the
    # default ten-second observation under shard load; keep the fault injector
    # installed until the owned operation actually finishes.
    store._owners.observation_timeout = 120
    application, resolver, values = await public_setup(store)
    initial = values[1]
    request = values[4].model_copy(update={"operation": initial.operation("000-request")})
    accepted = await application.accept_collaboration_request(request, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=initial.operation("001-control"),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    for index in range(observations):
        await application.register_collaboration_observation(
            accepted.expected,
            RequestObservation(
                key=f"observer-{index}",
                filter_commitment="all",
                projection_commitment="events",
                after_sequence=0,
                coverage_sequence=0,
                revision=1,
            ),
            context=resolver.context,
        )
    original = await application.inspect_collaboration_request(
        accepted.expected, context=resolver.context
    )
    assert len(original.event_sequences) == observations + 2
    _, rotated = await rotate(store, initial)
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
    pruning = NamespacePrune(
        operation=rotated.successor.reference.operation("first-prune"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before.retention_revision,
        max_records=batch,
    )
    result = await application.prune_collaboration_namespace(pruning, context=CONTEXT)
    assert result.removed_records == batch and not result.complete
    key = operation_key(accepted.expected.operation)
    async with store._transaction(initial.binding.application_scope, write=False) as tx:
        cursor = await tx.get("request_pruning", key)
        assert cursor is not None and cursor["next_event_index"] == batch
        assert await tx.get("requests", key) == original.model_dump(mode="json")
    with pytest.raises(CollaborationUnavailable):
        await application.inspect_collaboration_request(accepted.expected, context=resolver.context)

    reopened = stores()
    reopened._owners.observation_timeout = 120
    other = app(
        reopened,
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    assert await other.initialize_collaboration() == initial
    assert await other.prune_collaboration_namespace(pruning, context=CONTEXT) == result
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await tx.get("request_pruning", key) == cursor
        repository_type = type(tx)
    before_failure = await other.inspect_collaboration_namespace(context=CONTEXT)
    failed_batch = NamespacePrune(
        operation=rotated.successor.reference.operation("rollback-prune"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before_failure.retention_revision,
        # Remaining operation keys also include participant history. Give the
        # owner room to reach the request without imposing an artificial sort
        # order on those independently owned records.
        max_records=32,
    )
    delete = repository_type.delete
    deleted = []

    async def fail_after_event_delete(repository, table, record_key):
        await delete(repository, table, record_key)
        if table == "request_events":
            deleted.append(record_key)
            raise RuntimeError("injected post-deletion failure")

    with monkeypatch.context() as fault:
        fault.setattr(repository_type, "delete", fail_after_event_delete)
        with pytest.raises(CollaborationUnavailable):
            await other.prune_collaboration_namespace(failed_batch, context=CONTEXT)
    assert deleted == [(original.event_sequences[batch],)]
    assert await other.inspect_collaboration_namespace(context=CONTEXT) == before_failure
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await tx.get("request_pruning", key) == cursor
        assert await tx.get("requests", key) == original.model_dump(mode="json")
        event = await tx.get("request_events", deleted[0])
        assert event is not None
        operation = event["operation"]
        assert (
            await tx.get(
                "operations",
                (
                    operation["namespace_incarnation"],
                    operation["generation"],
                    operation["caller_key"],
                ),
            )
            is not None
        )
    retried = await other.prune_collaboration_namespace(failed_batch, context=CONTEXT)
    assert 1 <= retried.removed_records <= 32
    assert await other.prune_collaboration_namespace(failed_batch, context=CONTEXT) == retried
    final = retried
    for index in range(10):
        if final.complete:
            break
        before = await other.inspect_collaboration_namespace(context=CONTEXT)
        final = await other.prune_collaboration_namespace(
            NamespacePrune(
                operation=rotated.successor.reference.operation(f"next-prune-{index}"),
                namespace=rotated.namespace.reference,
                expected_retention_revision=before.retention_revision,
                max_records=32,
            ),
            context=CONTEXT,
        )
        if final.complete:
            break
    else:
        pytest.fail("Bounded request reclamation did not finish.")
    async with reopened._transaction(initial.binding.application_scope, write=False) as tx:
        assert await tx.get("request_pruning", key) is None
        assert await tx.get("requests", key) is None
        for sequence in original.event_sequences:
            assert await tx.get("request_events", (sequence,)) is None
