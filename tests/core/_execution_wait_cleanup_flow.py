"""Retire real foreign history before reconciling native wait cleanup."""

from tests.core.test_collaboration_namespace import rotate
from tests.core.test_participant_identity import CONTEXT, app

from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestControl
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def prune_and_reopen(
    application, source, store, stores, tmp_path, request, resolver, initialized, receipt, bound
):
    await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("cancel-before-pruning"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    _, rotated = await rotate(source, initialized)
    assert rotated.namespace.outstanding_obligations == 0, rotated.namespace
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire-before-wait-ack"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    for index in range(40):
        state = await application.inspect_collaboration_namespace(context=CONTEXT)
        if state.pruned_through >= rotated.namespace.reference.generation:
            break
        await application.prune_collaboration_namespace(
            NamespacePrune(
                operation=rotated.successor.reference.operation(f"prune-before-ack-{index}"),
                namespace=rotated.namespace.reference,
                expected_retention_revision=state.retention_revision,
                max_records=3,
            ),
            context=CONTEXT,
        )
    else:
        raise AssertionError("Wait history did not reach bounded pruning completion")
    assert await source.load_wait(initialized, bound, redactor=application._secret_redactor) is None
    registration = application._participant_coordinator._registration
    if not isinstance(source, InMemoryCollaborationStore):
        await source.close()
        source = stores()
        await store.close()
        if isinstance(store, SQLiteSessionStore):
            store = SQLiteSessionStore(tmp_path / "execution-wait.sqlite")
        else:
            store = PostgresSessionStore(request.getfixturevalue("postgres_dsn"))
    # Expired/revoked historical authority must not be needed on reconstruction.
    resolver.denied = True
    application = app(
        source,
        registration,
        session_store=store,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    assert await application.initialize_collaboration() == initialized
    return application, source, store
